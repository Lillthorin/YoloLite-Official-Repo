from torchvision.ops import box_iou, nms
import random, sys
import os
import numpy as np
import torch
import torch.nn as nn
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from scripts.helpers.utils_ms import decode_preds_anchorfree
import json, os


def yolo_collate(batch):
    # batch: list of (img_tensor, target_dict)
    imgs, targets = zip(*batch)
    # gör till listor (inte tuples) så vi kan manipla senare om vi vill
    return list(imgs), list(targets)

# =============== Hjälpare ===============
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def denormalize(img_tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    img = img_tensor.detach().cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img



def diou_nms(boxes, scores, iou_threshold=0.5):
    # bbox: [N,4] xyxy
    # scores: [N]
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        # lägg till DIoU-penalty
        cx1, cy1 = (boxes[i,0]+boxes[i,2])/2, (boxes[i,1]+boxes[i,3])/2
        cx2, cy2 = (boxes[idxs[1:],0]+boxes[idxs[1:],2])/2, (boxes[idxs[1:],1]+boxes[idxs[1:],3])/2
        center_dist = (cx1-cx2)**2 + (cy1-cy2)**2
        w = torch.max(boxes[i,2], boxes[idxs[1:],2]) - torch.min(boxes[i,0], boxes[idxs[1:],0])
        h = torch.max(boxes[i,3], boxes[idxs[1:],3]) - torch.min(boxes[i,1], boxes[idxs[1:],1])
        c2 = w**2 + h**2 + 1e-7
        dious = ious - center_dist/c2
        idxs = idxs[1:][dious <= iou_threshold]
    return torch.as_tensor(keep, device=boxes.device)


def _xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    """
    Tar xyxy i valfri shape. Returnerar [N,4] xywh.
    Tål tomma tensors och fel form (t.ex. (0,)).
    """
    if xyxy is None:
        return torch.zeros((0, 4), dtype=torch.float32)

    # Se till att vi har [..., 4]
    if xyxy.numel() == 0:
        # skapa korrekt tom tensor med 4 kolumner, behåll device/dtype om möjligt
        return xyxy.new_zeros((0, 4))

    # Om sista dimensionen inte är 4: försök reshape(-1, 4)
    if xyxy.dim() == 1 and xyxy.numel() == 4:
        xyxy = xyxy.view(1, 4)
    elif xyxy.shape[-1] != 4:
        assert xyxy.numel() % 4 == 0, f"xyxy måste vara multiplicerbart med 4, fick shape {tuple(xyxy.shape)}"
        xyxy = xyxy.view(-1, 4)

    x1, y1, x2, y2 = xyxy.unbind(-1)
    w = (x2 - x1).clamp_min(0)
    h = (y2 - y1).clamp_min(0)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


@torch.no_grad()
def _decode_batch_to_coco_dets(preds, img_size, conf_th=0.001, iou_th=0.65, add_one=True):
    """
    Returnerar lista[List[dict]] med COCO dets för varje bild i batchen.
    Använder utils_ms.decode_preds_anchorfree om du vill – men vi kan också dekoda direkt
    med din ritkod. Här tar vi den snabba vägen: vi förväntar oss redan 'preds' från modellen.
    """
    # Använd din befintliga decode-hjälpare om den finns:
    # decoded = decode_preds_anchorfree(preds, img_size=img_size, center_mode="v8", wh_mode="softplus")
    # För konsekvens med dina debug-bilder kör vi samma strategi som där:
    decoded = decode_preds_anchorfree(preds, img_size=img_size, center_mode="v8", wh_mode="softplus")

    B = decoded["box"].shape[0]
    C = decoded["cls"].shape[-1]
    out = [[] for _ in range(B)]

    for b in range(B):
        boxes  = decoded["box"][b]          # [N,4] xyxy i pixlar
        cls_lo = decoded["cls"][b]          # [N,C] logits
        obj_lo = decoded["obj"][b].squeeze(-1)  # [N] logits

        # YOLO-style score = sigmoid(obj) * max(sigmoid(cls))
        obj = obj_lo.sigmoid()
        cls_p = cls_lo.sigmoid()
        if C > 1:
            confs, clsi = cls_p.max(dim=-1)        # [N], [N]
            scores = obj * confs                   # [N]
        else:
            clsi   = torch.zeros_like(obj, dtype=torch.long)
            scores = obj

        # Svag tröskel före NMS (COCO-mAP kräver lågt conf_th för recall)
        keep0 = scores > conf_th
        if keep0.sum() == 0:
            continue
        boxes  = boxes[keep0]
        scores = scores[keep0]
        clsi   = clsi[keep0]

        # Per-klass NMS
        final_boxes, final_scores, final_cls = [], [], []
        for c in clsi.unique():
            m = (clsi == c)
            if m.sum() == 0:
                continue
            k = nms(boxes[m], scores[m], iou_th)
            if k.numel() == 0:
                continue
            final_boxes.append(boxes[m][k])
            final_scores.append(scores[m][k])
            final_cls.append(torch.full((k.numel(),), int(c.item()), device=boxes.device))
        if not final_boxes:
            continue

        fb = torch.cat(final_boxes,  dim=0)
        fs = torch.cat(final_scores, dim=0)
        fc = torch.cat(final_cls,    dim=0)

        # COCO: bbox i XYWH, category_id bör börja på 1
        bxywh = _xyxy_to_xywh(fb).cpu().tolist()
        sc    = fs.cpu().tolist()
        cc    = (fc + (1 if add_one else 0)).cpu().tolist()

        dets = []
        for bx, sc_, cid in zip(bxywh, sc, cc):
            dets.append({"category_id": int(cid), "bbox": [float(v) for v in bx], "score": float(sc_)})
        out[b] = dets
    return out

def _coco_eval_from_lists(coco_images, coco_anns, coco_dets, iouType="bbox", num_classes=None):
    """
    coco_images: [{"id":int,"file_name":str,"width":int,"height":int}, ...]
    coco_anns:   [{"id":int,"image_id":int,"category_id":int,"bbox":[x,y,w,h],"area":float,"iscrowd":0}, ...]
    coco_dets:   [{"image_id":int,"category_id":int,"bbox":[x,y,w,h],"score":float}, ...]
    """
    import os, json, tempfile
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # --- Guard 1: hantera tomma detections snyggt (pycocotools kraschar annars)
    if not coco_dets:
        # Bestäm antal klasser från GT om möjligt
        if num_classes is None:
            if coco_anns:
                max_cid = max(a["category_id"] for a in coco_anns)
                num_classes = int(max(1, max_cid))
            else:
                num_classes = 1
        # Returnera nollor – så träningsloopen kan fortsätta utan krasch
        return {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "APS": 0.0, "APM": 0.0, "APL": 0.0, "AR": 0.0
        }

    # --- Bestäm klasser (som tidigare)
    if num_classes is None:
        if len(coco_anns):
            max_cid = max(a["category_id"] for a in coco_anns)
            num_classes = int(max(1, max_cid))
        else:
            max_cid = max((d["category_id"] for d in coco_dets), default=1)
            num_classes = int(max(1, max_cid))

    categories = [{"id": i, "name": str(i)} for i in range(1, num_classes + 1)]

    # Säkert sätt att skriva temporära filer
    gt_fd, gt_path = tempfile.mkstemp(suffix=".json")
    dt_fd, dt_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(gt_fd, "w", encoding="utf-8") as fg:
            json.dump({
                "info": {"description": "Auto COCO GT", "version": "1.0"},
                "licenses": [],
                "images": coco_images if coco_images else [],
                "annotations": coco_anns,
                "categories": categories,
            }, fg)

        with os.fdopen(dt_fd, "w", encoding="utf-8") as fr:
            json.dump(coco_dets, fr)

        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(dt_path)
        E = COCOeval(coco_gt, coco_dt, iouType=iouType)
        E.evaluate(); E.accumulate(); E.summarize()
        return {
            "AP":   float(E.stats[0]),
            "AP50": float(E.stats[1]),
            "AP75": float(E.stats[2]),
            "APS":  float(E.stats[3]),
            "APM":  float(E.stats[4]),
            "APL":  float(E.stats[5]),
            "AR":   float(E.stats[8]),
            "ARS":  float(E.stats[9]),
            "ARM":  float(E.stats[10]),
            "ARL":  float(E.stats[11])
        }
    finally:
        try: os.remove(gt_path)
        except Exception: pass
        try: os.remove(dt_path)
        except Exception: pass

def _write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic på både Win/Linux

def _append_csv(path, header: list, row: list):
    make_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if make_header:
            f.write(",".join(header) + "\n")
        # allt som strängar
        f.write(",".join(str(x) for x in row) + "\n")



def yolo_collate(batch):
    # batch: list of (img_tensor, target_dict)
    imgs, targets = zip(*batch)
    # gör till listor (inte tuples) så vi kan manipla senare om vi vill
    return list(imgs), list(targets)



@torch.no_grad()
def save_val_debug_anchorfree(imgs, preds, epoch, out_dir,
                              img_size=416, conf_th=0.35, iou_th=0.60, topk=300, max_images=2,
                              mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                              center_mode="v8", wh_mode="softplus"):
    """
    Helt anchor-free rit-funktion. Ingen anchors_per_level och inga anchor-beroenden.
    Stöd för preds per nivå: list/tuple av tensors med shape [B,A,S,S,5+C] eller [B,S,S,5+C].
    """
    import os, cv2, torch, numpy as np
    import torch.nn.functional as F
    from torchvision.ops import nms as nms_iou

    os.makedirs(out_dir, exist_ok=True)
    device = imgs.device

    # Normalisera input till lista av nivåer
    preds_list = preds if isinstance(preds, (list, tuple)) else [preds]
    B = preds_list[0].shape[0]
    C = preds_list[0].shape[-1] - 5  # cls-dim

    # --- Hjälpfunktioner ---
    def _denorm(img_t):
        img = img_t.detach().float().cpu()
        # om bilden redan är 0..1-normaliserad med (img-mean)/std:
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        np_img = (img.permute(1,2,0).numpy() * 255.0).clip(0,255).astype("uint8")
        return cv2.cvtColor(np.ascontiguousarray(np_img), cv2.COLOR_RGB2BGR)

    def _xywh_to_xyxy_t(xywh):
        x, y, w, h = xywh.unbind(-1)
        return torch.stack([x - w*0.5, y - h*0.5, x + w*0.5, y + h*0.5], dim=-1)

    def _make_grid(S, device):
        gy, gx = torch.meshgrid(torch.arange(S, device=device),
                                torch.arange(S, device=device), indexing="ij")
        return gx.view(1,1,S,S), gy.view(1,1,S,S)

    def _nms(boxes, scores, iou_thr):
        return nms_iou(boxes, scores, iou_thr)

    # För-alloc per bild
    per_b_boxes = [[] for _ in range(B)]
    per_b_scores = [[] for _ in range(B)]
    per_b_cls = [[] for _ in range(B)]

    # ---- Decode varje nivå (anchor-free) ----
    for li, pred in enumerate(preds_list):
        # Tillåt både [B, A, S, S, D] och [B, S, S, D]
        if pred.dim() == 5:
            B_, A, S, _, D = pred.shape
        elif pred.dim() == 4:
            B_, S, _, D = pred.shape
            pred = pred.unsqueeze(1)  # [B, 1, S, S, D]
            A = 1
        else:
            raise ValueError(f"Pred shape ogiltig: {pred.shape}")

        assert B_ == B, "Batch-dimension måste vara lika på alla nivåer"
        stride = img_size / S
        gx, gy = _make_grid(S, device)

        tx, ty = pred[..., 0], pred[..., 1]
        tw, th = pred[..., 2], pred[..., 3]
        tobj   = pred[..., 4]
        tcls   = pred[..., 5:]

        # center (anchor-free)
        if center_mode == "v8":
            px = ((torch.sigmoid(tx)*2 - 0.5) + gx) * stride
            py = ((torch.sigmoid(ty)*2 - 0.5) + gy) * stride
        else:
            px = (torch.sigmoid(tx) + gx) * stride
            py = (torch.sigmoid(ty) + gy) * stride

        # width/height (anchor-free)
        if   wh_mode == "v8":
            pw = (torch.sigmoid(tw)*2).pow(2) * stride
            ph = (torch.sigmoid(th)*2).pow(2) * stride
        elif wh_mode == "softplus":
            pw = F.softplus(tw) * stride
            ph = F.softplus(th) * stride
        else:
            pw = tw.clamp(-4,4).exp() * stride
            ph = th.clamp(-4,4).exp() * stride

        obj = torch.sigmoid(tobj)
        if C > 0:
            cls_prob = torch.sigmoid(tcls)  # [B,A,S,S,C]
            if C > 1:
                confs, cls_idx = cls_prob.max(dim=-1)  # [B,A,S,S]
                scores = obj * confs
            else:
                # VIKTIG PATCH: även 1-klass ska använda klass-sannolikhet
                confs = cls_prob.squeeze(-1)           # [B,A,S,S]
                cls_idx = torch.zeros_like(obj, dtype=torch.long)
                scores  = obj * confs
        else:
            # Inga klasser? (bara objectness)
            cls_idx = torch.zeros_like(obj, dtype=torch.long)
            scores  = obj

        # Gör allt per-batch (robustare än "off"-räkning)
        for b in range(B):
            sc_b = scores[b]                 # [A,S,S]
            m_b = sc_b > conf_th
            if not m_b.any():
                continue

            bx = px[b][m_b]; by = py[b][m_b]
            bw = pw[b][m_b]; bh = ph[b][m_b]
            sc = sc_b[m_b]
            cc = cls_idx[b][m_b]

            # kasta bort extremt små boxar (kan annars “spamma” NMS)
            min_side = 2.0
            keep_sz = (bw >= min_side) & (bh >= min_side)
            if not keep_sz.any():
                continue

            bx = bx[keep_sz]; by = by[keep_sz]
            bw = bw[keep_sz]; bh = bh[keep_sz]
            sc = sc[keep_sz]; cc = cc[keep_sz]

            boxes_xyxy = _xywh_to_xyxy_t(torch.stack([bx,by,bw,bh], dim=1))
            boxes_xyxy[:,0::2] = boxes_xyxy[:,0::2].clamp(0, img_size-1)
            boxes_xyxy[:,1::2] = boxes_xyxy[:,1::2].clamp(0, img_size-1)

            per_b_boxes[b].append(boxes_xyxy)
            per_b_scores[b].append(sc)
            per_b_cls[b].append(cc)

    # ---- NMS & ritning per bild ----
    for b in range(min(B, max_images)):
        if len(per_b_boxes[b]) == 0:
            continue

        boxes_all  = torch.cat(per_b_boxes[b], dim=0)
        scores_all = torch.cat(per_b_scores[b], dim=0)
        cls_all    = torch.cat(per_b_cls[b],    dim=0)

        pre = int(boxes_all.shape[0])

        final_boxes, final_scores, final_cls = [], [], []
        for c in cls_all.unique():
            m = (cls_all == c)
            if m.sum() == 0:
                continue
            keep = _nms(boxes_all[m], scores_all[m], iou_th)
            if keep.numel() == 0:
                continue
            final_boxes.append(boxes_all[m][keep])
            final_scores.append(scores_all[m][keep])
            final_cls.append(torch.full((len(keep),), int(c), device=boxes_all.device))

        if len(final_boxes):
            boxes_k  = torch.cat(final_boxes)
            scores_k = torch.cat(final_scores)
            cls_k    = torch.cat(final_cls)
        else:
            boxes_k  = boxes_all[:0]
            scores_k = scores_all[:0]
            cls_k    = cls_all[:0]

        # extra conf-filter + top-k
        conf_mask = scores_k > conf_th
        boxes_k, scores_k, cls_k = boxes_k[conf_mask], scores_k[conf_mask], cls_k[conf_mask]
        if scores_k.numel() > topk:
            top_idx = scores_k.topk(topk).indices
            boxes_k, scores_k, cls_k = boxes_k[top_idx], scores_k[top_idx], cls_k[top_idx]

        post = int(boxes_k.shape[0])
        

        # ---- Rita ----
        img_np = _denorm(imgs[b])
        for box, score, cid in zip(boxes_k, scores_k, cls_k):
            x1,y1,x2,y2 = [int(round(v)) for v in box.tolist()]
            cv2.rectangle(img_np, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img_np, f"{int(cid)}:{float(score):.2f}", (x1, max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imwrite(os.path.join(out_dir, f"last_b{b}.jpg"), img_np)



