# train.py
import os, yaml, time, random, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from scripts.helpers.helpers import  _coco_eval_from_lists, _decode_batch_to_coco_dets, _xyxy_to_xywh
from scripts.data.p_r_f1 import build_curves_from_coco

# utils/summary_cards.py
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import math

Number = Union[int, float]

# vilka labels ska tolkas som procent (0..1 -> 0..100 %)
_PERCENT_LIKE = {"map", "ap", "ap50", "ap75", "aps", "apm", "apl",
                 "ar", "recall", "precision", "f1", "mAP@50", "mAP@50-95"}
# --- lägg detta någonstans ovanför evaluate_model (eller inuti, högst upp) ---
import time
from copy import deepcopy

@torch.no_grad()
def _bench_forward_ms_per_img(model: torch.nn.Module,
                              loader,
                              device: str,
                              use_amp: bool,
                              bench_batches: int = 5) -> float:
    """
    Mäter ren forward-tid (modell(imgs)) i ms per bild över 'bench_batches' batchar.
    Ingen decode/NMS/COCO-bygge – endast inference.
    """
    model.eval()
    it = iter(loader)
    # Warmup 2 batchar (utan timing) för stabilare cache/graph
    for _ in range(2):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader); imgs, _ = next(it)
        imgs = torch.stack(imgs).to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=(use_amp and device=='cuda')):
            _ = model(imgs)

    # Mät
    times = []
    counted_imgs = 0
    for _ in range(bench_batches):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader); imgs, _ = next(it)

        imgs = torch.stack(imgs).to(device, non_blocking=True)
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.amp.autocast(device_type='cuda', enabled=(use_amp and device=='cuda')):
            _ = model(imgs)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        B = imgs.size(0)
        times.append(dt_ms)
        counted_imgs += B

    if counted_imgs == 0:
        return float('nan')
    # ms per bild = (sum ms) / antal bilder
    return (sum(times) / max(1, counted_imgs))


def _format_ms(ms: float) -> tuple:
    """Returnerar (ms_per_img, fps) med tre decimalsiffror/FPS till 2."""
    if not (ms == ms) or ms <= 0:
        return (float('nan'), float('nan'))
    fps = 1000.0 / ms
    return (round(ms, 3), round(fps, 2))

def _load_font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _fmt_value(label: str, v: Number) -> str:
    lab = label.lower()
    # Procent: om label ser ut som ett procentmått och värdet är inom [0,1]
    if any(k in lab for k in _PERCENT_LIKE) and (0.0 <= float(v) <= 1.0000001):
        return f"{float(v)*100:.1f}%"
    # Förluster/övrigt
    if "loss" in lab:
        return f"{float(v):.3f}"
    if "conf" in lab or "iou" in lab or "lr" in lab:
        return f"{float(v):.3f}"
    return f"{float(v):.3f}"

def _draw_card(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], w: int, h: int,
               title: str, value: str, hint: str = ""):
    x, y = xy
    radius = 16
    # kortbakgrund
    draw.rounded_rectangle([x, y, x+w, y+h], radius=radius,
                           fill=(248,249,251), outline=(220,224,232), width=2)
    f_title = _load_font(24)
    f_value = _load_font(40)
    f_hint  = _load_font(16)

    draw.text((x+20, y+18), title, font=f_title, fill=(65,74,90))
    draw.text((x+20, y+62), value, font=f_value, fill=(33,37,41))
    if hint:
        draw.text((x+20, y+h-28), hint, font=f_hint, fill=(120,127,140))

def make_summary_image(
    metrics: List[Tuple[str, Number, Optional[str]]],
    title: str = "METRICS",
    subtitle: Optional[str] = None,
    cards_per_row: int = 3,
    card_size: Tuple[int,int] = (380, 140),
    gap: int = 20,
    pad: int = 24,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    metrics: list av (label, value, hint) – hint kan vara None.
             Ex: [("mAP@50", 0.957, ""), ("Precision", 0.953, ""), ("Recall", 0.930, "IoU=0.5")]
    """
    card_w, card_h = card_size
    n = len(metrics)
    rows = math.ceil(n / cards_per_row)
    W = pad*2 + cards_per_row*card_w + (cards_per_row-1)*gap
    H = pad*2 + rows*card_h + (rows-1)*gap + 70  # plats för rubrik

    img = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(img)

    # Rubrik-badge
    f_badge = _load_font(18)
    badge_w = max(120, int(draw.textlength(title, font=f_badge) + 24))
    draw.rounded_rectangle([pad, pad, pad+badge_w, pad+32],
                           radius=6, fill=(231,238,255))
    draw.text((pad+10, pad+7), title, font=f_badge, fill=(48,88,187))

    # Undertitel
    if subtitle:
        f_small = _load_font(16)
        draw.text((pad, pad+42), subtitle, font=f_small, fill=(120,127,140))

    # Rita kort
    ox, oy = pad, pad+70
    for i, (label, value, hint) in enumerate(metrics):
        cx, cy = i % cards_per_row, i // cards_per_row
        x = ox + cx*(card_w+gap)
        y = oy + cy*(card_h+gap)
        _draw_card(draw, (x,y), card_w, card_h, label, _fmt_value(label, value), hint or "")

    if save_path:
        img.save(save_path)
    return img

def evaluate_model(model, val_loader, log_dir, NUM_CLASSES, DEVICE, IMG_SIZE, batch_size):
    # -------------------- EVAL --------------------
    use_amp = True if torch.cuda.is_available() else False
    
    model_eval = model
    model_eval.eval()

    v_running = 0.0
    vb = vo = vc = 0.0

    # COCO-behållare för denna epoch
    coco_images, coco_anns, coco_dets = [], [], []
    ann_id = 1
    img_id = 1

    # (valfri) debug-index
    t = random.randrange(batch_size)

    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
        val_pbar = tqdm(enumerate(val_loader),
                        total=len(val_loader),
                        desc=f"evaluation",
                        leave=False)

        for i, (imgs, targets) in val_pbar:
            imgs = torch.stack(imgs).to(DEVICE, non_blocking=True)
            preds = model_eval(imgs)
            B = imgs.size(0)
            # Bygg COCO GT/DT
            
            batch_dets = _decode_batch_to_coco_dets(
                preds, img_size=IMG_SIZE, conf_th=0.001, iou_th=0.65, add_one=True
            )

            for b in range(B):
                coco_images.append({
                    "id": img_id,
                    "file_name": f"val_{img_id}.jpg",
                    "width": int(IMG_SIZE), "height": int(IMG_SIZE)
                })

                if "boxes" in targets[b] and targets[b]["boxes"] is not None:
                    gt_xyxy = targets[b]["boxes"]
                    if isinstance(gt_xyxy, np.ndarray):
                        gt_xyxy = torch.as_tensor(gt_xyxy)
                    gt_xywh = _xyxy_to_xywh(gt_xyxy)

                    gtl = targets[b].get("labels", None)
                    if gtl is None:
                        gtl = targets[b].get("classes", None)
                    if isinstance(gtl, np.ndarray):
                        gtl = torch.as_tensor(gtl)
                    if gtl is None:
                        gtl = torch.zeros((gt_xywh.size(0),), dtype=torch.long)

                    for bx, clsid0 in zip(gt_xywh.cpu().tolist(), gtl.cpu().tolist()):
                        coco_anns.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(clsid0) + 1,
                            "bbox": [float(v) for v in bx],
                            "area": float(max(0.0, bx[2]*bx[3])),
                            "iscrowd": 0,
                        })
                        ann_id += 1

                for d in batch_dets[b]:
                    coco_dets.append({
                        "image_id": img_id,
                        "category_id": int(d["category_id"]),
                        "bbox": [float(v) for v in d["bbox"]],
                        "score": float(d["score"]),
                    })

                img_id += 1
                
            

    # COCOeval 
    coco_stats = _coco_eval_from_lists(
        coco_images, coco_anns, coco_dets, iouType="bbox", num_classes=NUM_CLASSES
    )
    # Precision/Recall/F1-curves
    summary = build_curves_from_coco(
        coco_images=coco_images,
        coco_anns=coco_anns,
        coco_dets=coco_dets,
        out_dir=Path(log_dir) / f"curves",
        iou=0.50,
        steps=201
    )

    # -------------------- BENCHMARK: GPU + CPU --------------------
    bench_batches = 10  # justera vid behov (3–10 brukar räcka)

    # a) GPU (om finns)
    gpu_ms_per_img = float('nan'); gpu_fps = float('nan')
    if torch.cuda.is_available():
        try:
            gpu_ms = _bench_forward_ms_per_img(
                model=model_eval, loader=val_loader,
                device='cuda', use_amp=True, bench_batches=bench_batches
            )
            gpu_ms_per_img, gpu_fps = _format_ms(gpu_ms)
        except Exception as e:
            print(f"[bench][GPU] failed: {e}")

    # b) CPU – kopiera modell till CPU (påverkar inte originalet)
    cpu_ms_per_img = float('nan'); cpu_fps = float('nan')
    try:
        model_cpu = deepcopy(model_eval).to('cpu').eval()
        cpu_ms = _bench_forward_ms_per_img(
            model=model_cpu, loader=val_loader,
            device='cpu', use_amp=False, bench_batches=bench_batches
        )
        cpu_ms_per_img, cpu_fps = _format_ms(cpu_ms)
        del model_cpu
    except Exception as e:
        print(f"[bench][CPU] failed: {e}")

    plt.figure()
    plt.plot(summary["confs"], summary["P_curve"])
    plt.xlabel("Confidence threshold"); plt.ylabel("Precision")
    plt.title(f"Precision vs Confidence @ iou: 0.5")
    plt.grid(True, linestyle=":"); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "P_curve.png")); plt.close()

    plt.figure()
    plt.plot(summary["confs"], summary["R_curve"])
    plt.xlabel("Confidence threshold"); plt.ylabel("Recall")
    plt.title(f"Recall vs Confidence @ iou: 0.5")
    plt.grid(True, linestyle=":"); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "R_curve.png")); plt.close()

    plt.figure()
    plt.plot(summary["confs"], summary["F1_curve"])
    plt.xlabel("Confidence threshold"); plt.ylabel("F1")
    plt.title(f"F1 vs Confidence @ iou: 0.5")
    plt.grid(True, linestyle=":"); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "F1_curve.png")); plt.close()

    mAP50     = coco_stats["AP50"]
    precision = summary["precision_at_best"]
    recall    = summary["recall_at_best"]
    f1        = summary["best_f1"]
    ap        = coco_stats["AP"]
    ar        = coco_stats["AR"]
    best_conf = summary["best_conf"]
    aps       = coco_stats["APS"]
    apm       = coco_stats["APM"]
    apl       = coco_stats["APL"]
    ars       = coco_stats["ARS"]
    arm       = coco_stats["ARM"]
    arl       = coco_stats["ARL"]
    gpu_item = None
    if torch.cuda.is_available():
        gpu_item = (
            "Infer GPU (ms/img)",
            gpu_ms_per_img,  # float eller nan
            f"≈ {gpu_fps} FPS over {bench_batches} batches"
        )

    cpu_item = (
        "Infer CPU (ms/img)",
        cpu_ms_per_img,      # float eller nan
        f"≈ {cpu_fps} FPS over {bench_batches} batches"
    )
       # ---- bygg lista ----
    metrics_list = [
        ("mAP@50", mAP50, "mean Average Precision@50"),
        ("Precision", precision, "Precision at best F1"),
        ("Recall", recall, "Recall at best F1"),
        ("F1-score", f1, "Best F1 score"),
        ("mAP@50-95", ap, "mean Average Precision@50-95"),
        ("AR@0.50:0.95", ar, "Average Recall"),
        ("APS@0.50:0.95", aps, "Average Precision Small-objects"),
        ("APM@0.50:0.95", apm, "Average Precision Medium-objects"),
        ("APL@0.50:0.95", apl, "Average Precision Large-objects"),
        ("ARS@0.50:0.95", ars, "Average Recall Small-objects"),
        ("ARM@0.50:0.95", arm, "Average Recall Medium-objects"),
        ("ARL@0.50:0.95", arl, "Average Recall Large-objects"),
        ("Best conf", best_conf, "Threshold for highest F1"),
        gpu_item,
        cpu_item,
    ]
     # ---- sanering: ta bort None + fel form, och tvinga value->float ----
    def _normalize_metrics(items):
        out = []
        for m in items:
            if m is None or not isinstance(m, (list, tuple)) or len(m) != 3:
                continue
            label, val, hint = m
            try:
                val = float(val)
            except Exception:
                val = float('nan')
            out.append((str(label), val, "" if hint is None else str(hint)))
        return out

    metrics_list = _normalize_metrics(metrics_list)

    img = make_summary_image(
        metrics=metrics_list,
        title="METRICS",
        subtitle=f"• IoU 0.50 • Img-size {IMG_SIZE}",
        cards_per_row=3,
        save_path=f"{log_dir}\\summary.png",
    )
        
        
   
