import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, csv

def build_curves_from_coco(coco_images, coco_anns, coco_dets,
                           out_dir, iou=0.50, steps=201):
    """
    Bygger:
      - PR-kurva (rankad efter score)
      - Precision/Recall/F1 vs confidence (0..1)
    och sparar CSV + PNG i `out_dir`. Returnerar en summary-dict.

    Param:
      coco_images: [ { "id": int, "width": int, "height": int, ... }, ... ] (ej krav)
      coco_anns:   [ { "image_id": int, "category_id": int, "bbox": [x,y,w,h], ... }, ... ]
      coco_dets:   [ { "image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float }, ... ]
      out_dir:     str eller Path (skapas vid behov)
      iou:         IoU-tröskel för matchning (default 0.50)
      steps:       antal tröskelsteg mellan 0..1 för P/R/F1 (default 201)

    Obs:
      - Matchning sker PER KLASS (category_id), girigt mot o-matchade GT vid vald IoU.
      - För vettig PR-kurva: se till att dina `coco_dets` innehåller *låga* konfidenser också
        (t.ex. eval-decode med conf_th=0.001, per-class NMS, maxDets=100).
    """

    

    # ---------------- helpers ----------------
    def iou_xywh(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
        return inter / ua if ua > 0 else 0.0

    def build_gt_index(anns):
        # (img_id, cat_id) -> list of GT bboxes
        d = {}
        for a in anns:
            key = (int(a["image_id"]), int(a["category_id"]))
            d.setdefault(key, []).append(a["bbox"])
        return d

    # ---------------- PR curve (score-rankad svep) ----------------
    gt_index = build_gt_index(coco_anns)
    matched_flags = {k: np.zeros(len(v), dtype=bool) for k, v in gt_index.items()}
    total_gt = sum(len(v) for v in gt_index.values())

    dets_sorted = sorted(coco_dets, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    tps, fps = [], []
    for d in dets_sorted:
        key = (int(d["image_id"]), int(d["category_id"]))
        gts = gt_index.get(key, [])
        if len(gts) == 0:
            fps.append(1.0); tps.append(0.0)
            continue
        flags = matched_flags[key]
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if flags[j]:
                continue
            iou_val = iou_xywh(d["bbox"], g)
            if iou_val > best_iou:
                best_iou = iou_val; best_j = j
        if best_iou >= iou and best_j >= 0:
            flags[best_j] = True
            tps.append(1.0); fps.append(0.0)
        else:
            fps.append(1.0); tps.append(0.0)

    if len(tps) == 0:
        # inga prediktioner: skriv tomma filer och returnera
        return {
            "iou": float(iou),
            "best_f1": 0.0,
            "best_conf": 0.0,
            "precision_at_best": 0.0,
            "recall_at_best": 0.0,
            
        }

    tps = np.array(tps); fps = np.array(fps)
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls_rank = cum_tp / max(1, total_gt)
    precisions_rank = cum_tp / np.maximum(1, cum_tp + cum_fp)



    

    # ---------------- P/R/F1 vs confidence (0..1 svep) ----------------
    # För snabb åtkomst: indexera pred per (img, cat)
    det_index = {}
    for d in coco_dets:
        key = (int(d["image_id"]), int(d["category_id"]))
        det_index.setdefault(key, []).append(d)

    confs = np.linspace(0.0, 0.9, steps)
    P_curve, R_curve, F1_curve = [], [], []

    for thr in confs:
        TP = FP = 0
        matched = {k: np.zeros(len(v), dtype=bool) for k, v in gt_index.items()}
        for key, gts in gt_index.items():
            preds = [d for d in det_index.get(key, []) if float(d.get("score", 0.0)) >= thr]
            preds.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            flags = matched[key]
            for d in preds:
                best_j, best_iou = -1, 0.0
                for j, g in enumerate(gts):
                    if flags[j]:
                        continue
                    iou_val = iou_xywh(d["bbox"], g)
                    if iou_val > best_iou:
                        best_iou = iou_val; best_j = j
                if best_iou >= iou and best_j >= 0:
                    flags[best_j] = True
                    TP += 1
                else:
                    FP += 1
        FN = total_gt - TP
        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        P_curve.append(P); R_curve.append(R); F1_curve.append(F1)

    P_curve = np.array(P_curve); R_curve = np.array(R_curve); F1_curve = np.array(F1_curve)
    best_idx = int(np.argmax(F1_curve))
    # Välj din fasta tröskel (ex 0.50 eller 0.575)
    fixed_conf = 0.50
    idx = int(np.argmin(np.abs(confs - fixed_conf)))

    P_fixed = float(P_curve[idx])
    R_fixed = float(R_curve[idx])
    F1_fixed = float(F1_curve[idx])
    

    summary = {
        "iou": float(iou),
        "best_f1": float(F1_curve[best_idx]),
        "best_conf": float(confs[best_idx]),
        "precision_at_best": float(P_curve[best_idx]),
        "recall_at_best": float(R_curve[best_idx]),
        "fixed_conf": fixed_conf,
        "precision_at_fixed_conf": P_fixed,
        "recall_at_fixed_conf": R_fixed,
        "f1_at_fixed_conf": F1_fixed,
        "P_curve": P_curve,
        "R_curve": R_curve,
        "F1_curve": F1_curve,
        "confs": confs,
        "best_idx": best_idx
                
    }
    
    return summary
