# tools/infer.py
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch.nn as nn
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# =========================
# sys.path & imports
# =========================
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Modeller (samma som i train)
from scripts.model.model_v2 import YOLOLiteMS, YOLOLiteMS_CPU  # ändra om din väg skiljer

# Projektets decoder (om finns)
try:
    from scripts.helpers.utils_ms import decode_preds_anchorfree as project_decode  # din tränings-decode
    HAS_PROJECT_DECODER = True
except Exception:
    HAS_PROJECT_DECODER = False
    project_decode = None


# ========= build model from meta/config =========
def build_model_from_meta(meta: dict) -> nn.Module:
    cfg  = meta.get("config", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    tcfg = cfg.get("training", {}) or {}
    
    arch        = (meta.get("arch") or mcfg.get("arch") or "YOLOLiteMS").lower()
    backbone    = (meta.get("backbone") or mcfg.get("backbone") or "resnet18")
    num_classes = int(meta.get("num_classes") or mcfg.get("num_classes") or 80)
    num_anchors_per_level = tuple(meta.get("num_anchors_per_level") or (1,1,1))
    fpn_channels   = int(mcfg.get("fpn_channels", 128))
    depth_multiple = float(mcfg.get("depth_multiple", 1.0))
    width_multiple = float(mcfg.get("width_multiple", 1.0))
    head_depth     = int(mcfg.get("head_depth", 1))

    img_size = int(tcfg.get("img_size", meta.get("img_size", 640)))
    use_p6 = cfg["training"]["use_p6"]
    use_p2 = cfg["training"]["use_p2"]
    if arch == "yololitems":
        model = YOLOLiteMS(
            backbone=backbone,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            head_depth=head_depth,
            num_anchors_per_level=num_anchors_per_level,
            use_p6=use_p6,
            use_p2=use_p2
        )
    elif arch == "yololitems_cpu":
        model = YOLOLiteMS_CPU(
            backbone=backbone,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            head_depth=head_depth,
            num_anchors_per_level=num_anchors_per_level,
            use_p6=use_p6,
            use_p2=use_p2
        )
    else:
        raise ValueError(f"Okänd arch i meta/config: {arch}")
    return model


def load_model_names_imgsize_from_ckpt(weights: str, device: torch.device):
    """
    Laddar checkpoint av formatet {"state_dict","meta"}, bygger modell och returnerar:
    (model.eval på device, names:list[str], meta_img_size:int)
    """
    ckpt = torch.load(weights, map_location=device)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "meta" in ckpt):
        raise RuntimeError(
            "Checkpoint saknar 'state_dict'/'meta'. Spara vikter via save_checkpoint_state(...)."
        )

    meta = ckpt["meta"] or {}
    model = build_model_from_meta(meta)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[load_state_dict] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_state_dict] unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    names = meta.get("names") or [str(i) for i in range(int(meta.get("num_classes", 80)))]
    meta_img_size = int(meta.get("img_size", 640))
    return model, names, meta_img_size


# =========================
# Hjälpfunktioner (IO/NMS/ritning)
# =========================
def next_run_dir(base: str) -> str:
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        cand = root / str(n)
        try:
            cand.mkdir(parents=False, exist_ok=False)
            return str(cand.resolve())
        except FileExistsError:
            n += 1


def letterbox(im: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = im.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, scale, (left, top)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_th: float = 0.5, max_det: int = 300) -> torch.Tensor:
    try:
        from torchvision.ops import nms as tv_nms  # type: ignore
        keep = tv_nms(boxes, scores, iou_th)
    except Exception:
        # enkel klassagnostisk fallback
        keep_idx = []
        idxs = scores.argsort(descending=True)
        while idxs.numel() > 0:
            i = idxs[0]
            keep_idx.append(i)
            if idxs.numel() == 1:
                break
            ious = box_iou_single(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
            idxs = idxs[1:][ious <= iou_th]
        keep = torch.tensor(keep_idx, device=boxes.device)
    if keep.numel() > max_det:
        keep = keep[:max_det]
    return keep


def box_iou_single(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    a2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (a1 + a2 - inter + 1e-6)


# Lägg överst i filen:
import colorsys
from typing import Sequence

def _make_palette(n: int) -> list[tuple[int,int,int]]:
    """
    Stabil klasspalett (HSV → BGR). Ser ut som Ultralytics/YOLO-lika färger.
    """
    if n <= 0:
        return [(0, 255, 0)]
    hues = [i / max(1, n) for i in range(n)]
    cols = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)  # mättad, ljus
        cols.append((int(b*255), int(g*255), int(r*255)))  # BGR för cv2
    return cols

def _txt_size(text: str, font_scale: float = 0.5, thickness: int = 1):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return w, h + base

def draw_det(img_bgr: np.ndarray,
             boxes: np.ndarray,
             scores: np.ndarray,
             classes: np.ndarray,
             names: Sequence[str]) -> np.ndarray:
    """
    YOLO-lik overlay: klassfärger, fylld labelbakgrund, tunn AA-ram.
    """
    out = img_bgr.copy()
    H, W = out.shape[:2]
    n_classes = max(len(names), int(classes.max()+1) if classes.size else 0)
    palette = _make_palette(n_classes)

    # Tjocklek/skalning relativt bildstorlek
    t = max(1, int(round(0.002 * (H + W))))         # linjetjocklek
    fs = max(0.4, 0.0009 * (H + W))                 # font-scale

    for b, s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, b.tolist())
        cid = int(c)
        cls_name = names[cid] if 0 <= cid < len(names) else str(cid)
        color = palette[cid % len(palette)]

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t, lineType=cv2.LINE_AA)

        # Label text
        label = f"{cls_name} {s:.2f}"
        tw, th = _txt_size(label, font_scale=fs, thickness=max(1, t-1))
        # Bakgrundsremsa ovanför boxen (eller inuti om det är tight)
        bx1, by1 = x1, y1 - th - 3
        if by1 < 0:
            by1 = y1 + th + 3
        bx2, by2 = x1 + tw + 6, by1 + th + 2

        # Fylld label-bakgrund i klassfärg
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, -1, cv2.LINE_AA)

        # Text i svart eller vit beroende på bakgrundsluminans
        luminance = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
        txt_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)
        cv2.putText(out, label, (bx1 + 3, by2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, max(1, t-1), cv2.LINE_AA)
    return out


# =========================
# Fallback-decode (matchar din träning)
# =========================
def _make_grid(S: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    gy, gx = torch.meshgrid(torch.arange(S, device=device),
                            torch.arange(S, device=device), indexing="ij")
    return gx.float(), gy.float()   # [S,S]


def _xywh_to_xyxy_t(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(-1)
    return torch.stack([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], dim=-1)


def decode_anchorfree_like_train(
    preds: List[torch.Tensor],
    img_size: int,
    conf_th: float = 0.35,
    iou_th: float = 0.60,
    topk: int = 300,
    center_mode: str = "v8",
    wh_mode: str = "softplus",
) -> Dict[str, List[torch.Tensor]]:
    """
    Speglar save_val_debug_anchorfree: tar list/tuple av nivåer (eller en tensor),
    dekodar anchor-free med v8-center + softplus/v8-wh, kombinerar obj*cls,
    kör per-klass NMS och top-k. Returnerar pixel-boxar i padded kvadraten.
    """
    preds_list = preds if isinstance(preds, (list, tuple)) else [preds]
    B = preds_list[0].shape[0]
    device = preds_list[0].device

    out_boxes, out_scores, out_classes = [], [], []

    for b in range(B):
        boxes_all, scores_all, cls_all = [], [], []

        for pred in preds_list:
            # Tillåt [B,A,S,S,D] eller [B,S,S,D] (då A=1)
            if pred.dim() == 5:
                _, A, S, _, D = pred.shape
                p_b = pred[b]  # [A,S,S,D]
            elif pred.dim() == 4:
                _, S, _, D = pred.shape
                p_b = pred[b].unsqueeze(0)  # [1,S,S,D]
                A = 1
            else:
                raise ValueError(f"Pred shape ogiltig: {pred.shape}")

            gx, gy = _make_grid(S, device)

            tx = p_b[..., 0]; ty = p_b[..., 1]
            tw = p_b[..., 2]; th = p_b[..., 3]
            tobj = p_b[..., 4]
            tcls = p_b[..., 5:]
            C = tcls.shape[-1]

            cell = (img_size / S)
            # Center decode
            if center_mode == "v8":
                px = ((torch.sigmoid(tx) * 2.0 - 0.5) + gx) * cell   # [A,S,S]
                py = ((torch.sigmoid(ty) * 2.0 - 0.5) + gy) * cell
            else:
                px = (torch.sigmoid(tx) + gx) * cell
                py = (torch.sigmoid(ty) + gy) * cell

            # W/H decode
            if   wh_mode == "v8":
                pw = (torch.sigmoid(tw) * 2).pow(2) * cell
                ph = (torch.sigmoid(th) * 2).pow(2) * cell
            elif wh_mode == "softplus":
                pw = F.softplus(tw) * cell
                ph = F.softplus(th) * cell
            else:
                pw = tw.clamp(-4, 4).exp() * cell
                ph = th.clamp(-4, 4).exp() * cell

            obj = torch.sigmoid(tobj)

            if C > 0:
                cls_prob = torch.sigmoid(tcls)       # [A,S,S,C]
                if C > 1:
                    confs, cls_idx = cls_prob.max(dim=-1)  # [A,S,S]
                    scores = obj * confs
                else:
                    confs = cls_prob.squeeze(-1)     # [A,S,S]
                    cls_idx = torch.zeros_like(obj, dtype=torch.long)
                    scores = obj * confs
            else:
                cls_idx = torch.zeros_like(obj, dtype=torch.long)
                scores  = obj

            m = scores > conf_th
            if not m.any():
                continue

            bx = px[m]; by = py[m]
            bw = pw[m]; bh = ph[m]
            sc = scores[m]; cc = cls_idx[m]

            # Släng bort pyttesmå lådor
            min_side = 2.0
            keep_sz = (bw >= min_side) & (bh >= min_side)
            if not keep_sz.any():
                continue
            bx = bx[keep_sz]; by = by[keep_sz]
            bw = bw[keep_sz]; bh = bh[keep_sz]
            sc = sc[keep_sz]; cc = cc[keep_sz]

            boxes_xyxy = _xywh_to_xyxy_t(torch.stack([bx, by, bw, bh], dim=1))
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, img_size - 1)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, img_size - 1)

            boxes_all.append(boxes_xyxy)
            scores_all.append(sc)
            cls_all.append(cc)

        if len(boxes_all) == 0:
            out_boxes.append(torch.zeros((0, 4), device=device))
            out_scores.append(torch.zeros((0,), device=device))
            out_classes.append(torch.zeros((0,), device=device, dtype=torch.long))
            continue

        boxes_all  = torch.cat(boxes_all,  dim=0)
        scores_all = torch.cat(scores_all, dim=0)
        cls_all    = torch.cat(cls_all,    dim=0)

        # Per-klass NMS
        final_boxes, final_scores, final_cls = [], [], []
        for c in cls_all.unique():
            m = (cls_all == c)
            if m.sum() == 0:
                continue
            keep = nms(boxes_all[m], scores_all[m], iou_th)
            if keep.numel():
                final_boxes.append(boxes_all[m][keep])
                final_scores.append(scores_all[m][keep])
                final_cls.append(torch.full((keep.numel(),), int(c), device=device, dtype=torch.long))

        if len(final_boxes):
            boxes_k  = torch.cat(final_boxes)
            scores_k = torch.cat(final_scores)
            cls_k    = torch.cat(final_cls)
            # extra top-k
            if scores_k.numel() > topk:
                top_idx = scores_k.topk(topk).indices
                boxes_k, scores_k, cls_k = boxes_k[top_idx], scores_k[top_idx], cls_k[top_idx]
        else:
            boxes_k  = boxes_all[:0]
            scores_k = scores_all[:0]
            cls_k    = cls_all[:0]

        out_boxes.append(boxes_k)
        out_scores.append(scores_k)
        out_classes.append(cls_k)

    return {"boxes": out_boxes, "scores": out_scores, "classes": out_classes}


# =========================
# Huvudflöde
# =========================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path till checkpoint (.pt/.pth) som sparats med save_checkpoint_state(...)")
    ap.add_argument("--img", default=None, help="En bild att köra på")
    ap.add_argument("--img_dir", default=None, help="Eller en mapp med bilder (jpg/png)")
    ap.add_argument("--img_size", type=int, default=0, help="Override av meta.img_size (0 = använd meta)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--save_txt", action="store_true", help="Spara YOLO-format txt per bild")
    ap.add_argument("--no_letterbox", action="store_true", help="Använd ren resize istället för letterbox")
    args = ap.parse_args()

    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # Ladda modell/metadata
    model, names, meta_img_size = load_model_names_imgsize_from_ckpt(args.weights, device)
    effective_img_size = int(args.img_size) if int(args.img_size) > 0 else int(meta_img_size)

    # Inputlistor
    if args.img and Path(args.img).exists():
        paths = [args.img]
    elif args.img_dir and Path(args.img_dir).exists():
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = [str(p) for p in Path(args.img_dir).glob("*") if p.suffix.lower() in exts]
        paths.sort()
    else:
        raise ValueError("Ange --img eller --img_dir som existerar.")

    # Run-dir
    run_dir = next_run_dir("runs/infer")
    (Path(run_dir) / "labels").mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "json").mkdir(parents=True, exist_ok=True)

    # Preproc: matcha val (mean/std)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for pth in paths:
        img0 = cv2.imread(pth)
        if img0 is None:
            print(f"Varnar: kunde inte läsa {pth}")
            continue

        # Resize → kvadrat
        if args.no_letterbox:
            resized = cv2.resize(img0, (effective_img_size, effective_img_size), interpolation=cv2.INTER_LINEAR)
            lb, scale, (padx, pady) = resized, min(effective_img_size/img0.shape[0], effective_img_size/img0.shape[1]), (0, 0)
        else:
            lb, scale, (padx, pady) = letterbox(img0, new_size=effective_img_size)

        # Normalisera som i val
        im = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = (im - MEAN) / STD
        im = np.transpose(im, (2, 0, 1))  # CHW
        im = np.expand_dims(im, 0)        # NCHW
        im_t = torch.from_numpy(im).to(device)

        # Forward
        out = model(im_t)
        preds = out if isinstance(out, (list, tuple)) else [out]

        # Decode
        if HAS_PROJECT_DECODER:
            # Din tränings-decode: ger xyxy pixlar i padded kvadrat + logits
            decoded = project_decode(preds, img_size=effective_img_size, center_mode="v8", wh_mode="softplus")
            boxes_t = decoded["box"][0]  # [N,4] xyxy i padded kvadraten
            obj_log = decoded["obj"][0].squeeze(-1)  # [N]
            cls_log = decoded["cls"][0]              # [N,C]
            obj = obj_log.sigmoid()
            if cls_log.shape[-1] > 1:
                confs, cls_idx = cls_log.sigmoid().max(dim=-1)
                scores_t = obj * confs
            else:
                cls_idx = torch.zeros_like(obj, dtype=torch.long)
                scores_t = obj
            # per-klass NMS
            m0 = scores_t > args.conf
            boxes_t = boxes_t[m0]; scores_t = scores_t[m0]; classes_t = cls_idx[m0]
            final_boxes, final_scores, final_cls = [], [], []
            for c in classes_t.unique():
                mc = (classes_t == c)
                if mc.sum() == 0:
                    continue
                keep = nms(boxes_t[mc], scores_t[mc], args.iou)
                if keep.numel():
                    final_boxes.append(boxes_t[mc][keep])
                    final_scores.append(scores_t[mc][keep])
                    final_cls.append(torch.full((keep.numel(),), int(c), device=boxes_t.device, dtype=torch.long))
            if len(final_boxes):
                boxes_t  = torch.cat(final_boxes)
                scores_t = torch.cat(final_scores)
                classes_t= torch.cat(final_cls)
            else:
                boxes_t  = boxes_t[:0]
                scores_t = scores_t[:0]
                classes_t= classes_t[:0]
        else:
            # Fallback som matchar din träningslogik
            decoded = decode_anchorfree_like_train(
                preds,
                img_size=effective_img_size,
                conf_th=args.conf,
                iou_th=args.iou,
                topk=args.max_det,
                center_mode="v8",
                wh_mode="softplus",
            )
            boxes_t, scores_t, classes_t = decoded["boxes"][0], decoded["scores"][0], decoded["classes"][0]

        # --- Back-map till original ---
        boxes_px = boxes_t.clone()
        boxes_px[:, [0, 2]] -= padx
        boxes_px[:, [1, 3]] -= pady
        boxes_px /= max(scale, 1e-6)

        boxes_np = boxes_px.cpu().numpy()
        h0, w0 = img0.shape[:2]
        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, w0 - 1)
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, h0 - 1)

        scores_np = scores_t.detach().float().cpu().numpy() if scores_t.numel() else np.zeros((0,), dtype=np.float32)
        classes_np = classes_t.detach().long().cpu().numpy() if classes_t.numel() else np.zeros((0,), dtype=np.int64)

        # Rita & spara
        vis = draw_det(img0, boxes_np, scores_np, classes_np, names)
        out_path = str(Path(run_dir) / (Path(pth).stem + "_pred.jpg"))
        cv2.imwrite(out_path, vis)

        # YOLO-txt (xywh norm)
        if args.save_txt and boxes_np.size > 0:
            h, w = img0.shape[:2]
            xyxy = boxes_np
            cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0 / w
            cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0 / h
            bw = (xyxy[:, 2] - xyxy[:, 0]) / w
            bh = (xyxy[:, 3] - xyxy[:, 1]) / h
            txt = Path(run_dir) / "labels" / f"{Path(pth).stem}.txt"
            with open(txt, "w", encoding="utf-8") as f:
                for c, x, y, ww, hh, sc in zip(classes_np, cx, cy, bw, bh, scores_np):
                    f.write(f"{int(c)} {x:.6f} {y:.6f} {ww:.6f} {hh:.6f} {sc:.4f}\n")

        # JSON med resultat
        rec = []
        for b, s, c in zip(boxes_np.tolist(), scores_np.tolist(), classes_np.tolist()):
            rec.append({
                "bbox_xyxy": [float(x) for x in b],
                "score": float(s),
                "class_id": int(c),
                "class_name": names[int(c)] if int(c) < len(names) else str(int(c))
            })
        with open(Path(run_dir) / "json" / f"{Path(pth).stem}.json", "w", encoding="utf-8") as f:
            json.dump({"image": pth, "detections": rec}, f, ensure_ascii=False, indent=2)

        print(f"✓ Sparat: {out_path}")

    print(f"Allt sparat i: {run_dir}")


if __name__ == "__main__":
    main()


