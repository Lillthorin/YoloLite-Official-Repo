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
                
            

    # COCOeval för epoken (tyst, men en rad sammanfattning via tqdm.write)
    coco_stats = _coco_eval_from_lists(
        coco_images, coco_anns, coco_dets, iouType="bbox", num_classes=NUM_CLASSES
    )

    summary = build_curves_from_coco(
        coco_images=coco_images,
        coco_anns=coco_anns,
        coco_dets=coco_dets,
        out_dir=Path(log_dir) / f"curves",
        iou=0.50,
        steps=201
    )


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
        
        
   
