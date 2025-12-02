# train.py
import os, yaml, time, random, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random
from tqdm.auto import tqdm
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from torch.utils.data import DataLoader
from scripts.model.model_v2 import YOLOLiteMS, YOLOLiteMS_CPU
from scripts.data.dataset import YoloDataset
from scripts.data.augment import get_base_transform, get_val_transform
from scripts.loss.loss import LossAF
from scripts.helpers.sanity_check import visualize_batch
from scripts.helpers.schedulers import build_scheduler
from scripts.helpers.helpers import yolo_collate, _coco_eval_from_lists,  set_seed, save_val_debug_anchorfree, _decode_batch_to_coco_dets, _xyxy_to_xywh, _write_json_atomic, _append_csv
from scripts.args.build_args import build_argparser, load_configs, apply_overrides
from scripts.data.plot_metrics import plot_metrics
from scripts.helpers.evaluate import evaluate_model

def save_checkpoint_state(model, metrics: dict, class_names, config: dict, out_path: str):
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    meta = {
        "metric_key": metric_key,
        "metric_value": float(metrics.get(metric_key, -1.0)),
        "names": list(class_names) if class_names else None,
        "num_classes": int(config["model"]["num_classes"]),
        "img_size": int(config["training"].get("img_size", 640)),
        "arch": config["model"]["arch"],
        "backbone": config["model"]["backbone"],
        "config": config,  
    }
    torch.save({"state_dict": cpu_state, "meta": meta}, out_path)

def _build_num_anchors(use_p6, use_p2):
    if use_p2 and use_p6:
        return (1, 1, 1, 1, 1)
    if use_p2 or use_p6:
        return (1, 1, 1, 1)
    else:
        return (1, 1, 1)
def plot_metric_vs_conf(x, y, title, ylabel, best_idx, fixed_conf, out_path):
    import numpy as np, matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, label=title, linewidth=2)
    # markera "best F1" och "fixed_conf"
    if best_idx is not None:
        plt.axvline(float(x[best_idx]), linestyle="--", alpha=0.6, label=f"best @ {float(x[best_idx]):.3f}")
    plt.xlabel("Confidence")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# =============== Main ===============
if __name__ == "__main__":
    ap = build_argparser()
    opt = ap.parse_args()

    # 1) Läs & slå ihop configs
    config = load_configs(model_yaml=opt.model, train_yaml=opt.train, data_yaml=opt.data)

    # 2) Applicera CLI overrides (får sista ordet)
    config = apply_overrides(config, opt)

    # 3) Seed
    set_seed(config["training"].get("seed", 1337))

    # 4) (valfritt) skriv ut sammanställd config till log_dir för spårbarhet
    log_dir = config["logging"].get("log_dir", "runs/default")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(log_dir) / "merged_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    _DEVICE = config["training"]["device"]
    DEVICE = f"cuda:{_DEVICE}" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True if DEVICE == "cuda" else False

    NUM_CLASSES = int(config["model"]["num_classes"])
    IMG_SIZE = int(config["training"]["img_size"])
    use_augment = bool(config["training"]["augment"])   
    # Datasets & loaders
    val_resize = 1.0 if config["training"]["resize"] else 0.0
    train_resize = 1.0 if config["training"]["resize"] else 0.3
    train_ds = YoloDataset(
        config["dataset"]["train_images"],
        config["dataset"]["train_labels"],
        img_size=IMG_SIZE,
        is_train=True if use_augment else False,
        transforms=get_base_transform(IMG_SIZE, train_resize) if use_augment else get_val_transform(IMG_SIZE)
    )
    val_ds = YoloDataset(
        config["dataset"]["val_images"],
        config["dataset"]["val_labels"],
        img_size=IMG_SIZE,
        is_train=False,
        transforms=get_val_transform(IMG_SIZE, val_resize)
    )
    

    nw = int(config["training"].get("num_workers", 4))
    train_loader = DataLoader(
    train_ds,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=nw,
    pin_memory=True,
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
    collate_fn=yolo_collate,         # <-- INTE lambda
    drop_last=False,
)
    val_loader = DataLoader(
            val_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=yolo_collate,         # <-- INTE lambda
            drop_last=False,
        )
    
    

    batch_size = config["training"]["batch_size"]
    num_anchors_per_level = _build_num_anchors(config["training"]["use_p6"], config["training"]["use_p2"])
    if config["model"]["arch"].lower() == 'yololitems':              
        # --- Modell ---
        model = YOLOLiteMS(
            backbone=config["model"]["backbone"],
            num_classes=NUM_CLASSES,
            fpn_channels=config["model"]["fpn_channels"],
            width_multiple=config["model"].get("width_multiple", 1.0),
            depth_multiple=config["model"].get("depth_multiple", 1.0),
            head_depth=config["model"].get("head_depth", 1),
            num_anchors_per_level=num_anchors_per_level,   # t.ex. (3,3,3)
            use_p6=config["training"]["use_p6"],
            use_p2=config["training"]["use_p2"]
            
        ).to(DEVICE)
    elif config["model"]["arch"].lower() == 'yololitems_cpu':
        model = YOLOLiteMS_CPU(
            backbone=config["model"]["backbone"],
            num_classes=NUM_CLASSES,
            fpn_channels=config["model"]["fpn_channels"],
            num_anchors_per_level= num_anchors_per_level, # t.ex. (3,3,3)
            depth_multiple=config["model"].get("depth_multiple", 1.0),
            width_multiple=config["model"].get("width_multiple", 1.0),
            head_depth=config["model"].get("head_depth", 1),
            use_p6=config["training"]["use_p6"],
            use_p2 = config["training"]["use_p2"]
            
        ).to(DEVICE)

    # --- Loss ---
  
    criterion = LossAF(
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,

    # --- weights & toggles ---
    lambda_box=config.get("lambda_box", 5.0),
    lambda_obj=config.get("lambda_obj", 1.0),
    lambda_cls=config.get("lambda_cls", 1.0),
    focal=config.get("focal", False),
    gamma=config.get("gamma", 2.0),
    alpha=config.get("alpha", 0.25),
    cls_smoothing=config.get("cls_smoothing", 0.05),

    # --- matcher/assigner ---
    assign_cls_weight=config.get("assign_cls_weight", 0.5),
    center_radius_cells=config.get("center_radius_cells", 2.0),
    topk_limit=config.get("topk_limit", 20),

    # --- scale/area guards ---
    area_cells_min=config.get("area_cells_min", 4),
    area_cells_max=config.get("area_cells_max", 256),
    area_tol=config.get("area_tol", 1.25),

    # --- priors & costs ---
    size_prior_w=config.get("size_prior_w", 0.2),
    ar_prior_w=config.get("ar_prior_w", 0.1),
    iou_cost_w=config.get("iou_cost_w", 3.0),
    center_cost_w=config.get("center_cost_w", 0.5),
)
   
    # ---- safe float helper ----
    def f(x, default=None):
        if x is None:
            return default
        return float(x)

    # ---- läs och casta från config ----
    base_lr = f(config["training"]["lr"])                 # t.ex. "0.001" -> 0.001
    wd      = f(config["training"].get("weight_decay", 1e-4))

    bb_mult   = f(config["training"].get("bb_lr_mult", 1.0))
    neck_mult = f(config["training"].get("neck_lr_mult", 1.0))
    head_mult = f(config["training"].get("head_lr_mult", 1.0))

    lr_bb   = base_lr * bb_mult
    lr_neck = base_lr * neck_mult
    lr_head = base_lr * head_mult
    print(f"LRs: base {base_lr}, bb {lr_bb}, neck {lr_neck}, head {lr_head}")
    # ---- bygg parametergupper ----
    backbone_params = list(model.backbone.parameters())

    head_params = []
    for hn in ["head", "head3", "head4", "head5"]:
        if hasattr(model, hn):
            head_params += list(getattr(model, hn).parameters())

    collected = {id(p) for p in backbone_params + head_params}
    neck_params = [p for p in model.parameters() if id(p) not in collected]

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_bb,   "weight_decay": wd},
        {"params": [p for p in neck_params     if p.requires_grad], "lr": lr_neck, "weight_decay": wd},
        {"params": [p for p in head_params     if p.requires_grad], "lr": lr_head, "weight_decay": wd},
    ]
    # ----- Multiscale training ----
    multi_scale_sizes= config["training"].get("multi_scale_sizes", None)
    # ---- optimizer ----
    opt_name = str(config["training"].get("optimizer", "adamw")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.AdamW(param_groups)

    # ---- scheduler ----
    steps_per_epoch = max(1, len(train_loader))
    scheduler, sched_type = build_scheduler(optimizer, config, steps_per_epoch)


    use_amp = bool(config["training"].get("amp", True)) and (DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    grad_clip = float(config["training"].get("grad_clip", 0.0))
    save_every = int(config["training"].get("save_every", 10))
    log_dir = config["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    images, targets = next(iter(train_loader))
    visualize_batch(images, targets, save_path=os.path.join(log_dir, "sanity_check.jpg"))
    if config["training"]["resume"] is not None:
        ckpt = torch.load(config["training"]["resume"], map_location=DEVICE) 
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print("missing:", len(missing), "unexpected:", len(unexpected))
    print(model)
    print(f"Starting training on {DEVICE}, {len(train_ds)} train images, {len(val_ds)} val images, img-size: {IMG_SIZE}")
    best_val = float(0.000)
    train_losses, val_losses, mAP, F1, Recall, Precision, best_conf  = [], [], [], [], [], [], []
    Precision_fixed, Recall_fixed, F1_fixed = [], [], []
    epochs = int(config["training"]["epochs"])
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    if warmup_epochs > 0 and sched_type != "onecycle":
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * 0.1
    

    #Define save by;
    save_by = config["training"]["save_by"]

    
    class_names = config["dataset"]["names"]
    weight_folder = os.path.join(log_dir, 'weights')
    os.makedirs(weight_folder, exist_ok=True)
    best_ckpt_path = os.path.join(weight_folder, "best_model_state.pt")
    last_ckpt_path = os.path.join(weight_folder, "last_model_state.pt")
    best_no_aug = os.path.join(weight_folder, "best_no_aug.pt")
    metric_key = save_by  # eller "AP" om du vill optimera på COCO AP
    best_metric = -1.0
    best_metric_no_aug = -1.0
    val_thresh = 0.3
    for epoch in range(epochs):
        if epoch == (int(epochs*0.7)) and use_augment:
            train_ds.is_train = False
        if epoch > (int(epochs*0.9)):
            train_ds.transforms = get_val_transform(IMG_SIZE)
            use_augment = False
            train_ds.is_train = False
            
        # -------------------- TRAIN --------------------
        model.train()
        start = time.time()
        running = 0.0
        box_m = obj_m = cls_m = 0.0

        train_pbar = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f"Train {epoch+1}/{epochs}",
                        leave=False)

        for i, (imgs, targets) in train_pbar:
            imgs = torch.stack(imgs).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                preds = model(imgs)
                loss, loss_dict = criterion(preds, targets)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # statistik per batch (utan dubbelberäkning)
            B = imgs.size(0)
            box_show = float(loss_dict['box']) / B
            obj_show = float(loss_dict['obj']) / B
            cls_show = float(loss_dict['cls']) / B
            loss_show = box_show + obj_show + cls_show

            running += loss_show
            box_m   += box_show
            obj_m   += obj_show
            cls_m   += cls_show

            train_pbar.set_postfix(loss=f"{running/(i+1):.4f}",
                                box=f"{box_m/(i+1):.4f}",
                                obj=f"{obj_m/(i+1):.4f}",
                                cls=f"{cls_m/(i+1):.4f}")

        avg_train = running / max(1, len(train_loader))
        train_losses.append(avg_train)

        # -------------------- LR SCHEUDLING --------------------
        if sched_type == "onecycle":
            pass
        elif warmup_epochs > 0 and epoch < warmup_epochs and sched_type != "reduceonplat":
            w = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * (0.1 + 0.9 * w)
        elif scheduler is not None and sched_type not in ("reduceonplat", "onecycle"):
            scheduler.step()

        # -------------------- EVAL --------------------
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
                            desc=f"Val   {epoch+1}/{epochs}",
                            leave=False)

            for i, (imgs, targets) in val_pbar:
                imgs = torch.stack(imgs).to(DEVICE, non_blocking=True)
                preds = model_eval(imgs)

                vloss, vdict = criterion(preds, targets)
                B = imgs.size(0)
                
                vb += float(vdict['box']) / B
                vo += float(vdict['obj']) / B
                vc += float(vdict['cls']) / B
                v_running = vb + vo + vc
                if i == t and epoch+1 > 5:
                    save_val_debug_anchorfree(
                        imgs, preds, epoch, out_dir=log_dir,
                        img_size=IMG_SIZE, conf_th=val_thresh, iou_th=0.3,
                        topk=300, center_mode="v8", wh_mode="softplus"
                    )

                # Bygg COCO GT/DT
                
                batch_dets = _decode_batch_to_coco_dets(
                    preds, img_size=IMG_SIZE, conf_th=0.1, iou_th=0.65, add_one=True
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
                    if scheduler is not None and sched_type == "step":
                        scheduler.step()
                val_pbar.set_postfix(loss=f"{v_running/(i+1):.4f}",
                                    box=f"{vb/(i+1):.4f}",
                                    obj=f"{vo/(i+1):.4f}",
                                    cls=f"{vc/(i+1):.4f}")
                
        # Evaluate every : if epoch % 5 == 0:
        avg_val = v_running / max(1, len(val_loader))
        val_losses.append(avg_val)

        # COCOeval för epoken (tyst, men en rad sammanfattning via tqdm.write)
        coco_stats = _coco_eval_from_lists(
            coco_images, coco_anns, coco_dets, iouType="bbox", num_classes=NUM_CLASSES
        )
        coco_images, coco_anns, coco_dets = [], [], []
        elapsed = time.time() - start
       
        # --------- Loggning till filer (ingen print-spam) ----------
        metrics_csv = os.path.join(log_dir, "metrics.csv")
        metrics_last_json = os.path.join(log_dir, "last_metrics.json")
        metrics_best_json = os.path.join(log_dir, "best_metrics.json")

        lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

        csv_header = [
            "epoch","AP","AP50","AP75","APS","APM","APL","AR",
            "train_loss","val_loss","lr_g0","lr_g1","lr_g2","elapsed_s","timestamp"
        ]
        csv_row = [
            epoch+1,
            coco_stats["AP"], coco_stats["AP50"], coco_stats["AP75"],
            coco_stats["APS"], coco_stats["APM"], coco_stats["APL"], coco_stats["AR"],
            avg_train, avg_val,
            *(lrs + [None, None, None])[:3],
            elapsed, now_iso
        ] 
        
        _append_csv(metrics_csv, csv_header, csv_row)
        
        


        if scheduler is not None and sched_type == "reduceonplat":
            scheduler.step(avg_val)

        # The code snippet provided is written in Python and performs the following actions:
        
        current = coco_stats[metric_key]
        if (current > best_metric) and use_augment:
            best_metric = current
            # Spara EN fil: state_dict + metadata. Undvik torch.save(model)
            model_eval_cpu = model_eval.to("cpu").eval()
            save_checkpoint_state(model_eval_cpu, coco_stats, class_names, config, best_ckpt_path)
            model_eval.to(DEVICE).eval()
            print(f"✓ New best {metric_key}={best_metric:.4f} saved to {best_ckpt_path}")
           
            
        #Save best model without augmentation
        if (current > best_metric_no_aug) and not use_augment:
            best_metric_no_aug = current
            # Spara EN fil: state_dict + metadata. Undvik torch.save(model)
            model_eval_cpu = model_eval.to("cpu").eval()
            save_checkpoint_state(model_eval_cpu, coco_stats, class_names, config, best_no_aug)
            model_eval.to(DEVICE).eval()
            print(f"✓ New best {metric_key}={best_metric:.4f} saved to {best_no_aug}")   
    
        # Losskurva (tyst om den misslyckas)
        try:
  
            #Loss
            plt.figure()
            plt.plot(train_losses, label="Train")
            plt.plot(val_losses, label="Val")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve")
            plt.savefig(os.path.join(log_dir, "loss_curve.png")); plt.close()
            

        
        except Exception:
            pass
            #Save every x epoch
        if (epoch+1) % save_every == 0:
            save_path = os.path.join(weight_folder, f"epoch_{epoch+1}.pt")
            model_eval_cpu = model_eval.to("cpu").eval()
            save_checkpoint_state(model_eval_cpu, coco_stats, class_names, config, save_path)   
            model_eval.to(DEVICE).eval()
        model_eval_cpu = model_eval.to("cpu").eval()
        save_checkpoint_state(model_eval_cpu, coco_stats, class_names, config, last_ckpt_path)
        model_eval.to(DEVICE).eval()        
        # En enda kort sammanfattningsrad per epoch
        
        tqdm.write(
            f"Epoch {epoch+1}/{epochs} | "
            f"train {avg_train:.4f} | val {avg_val:.4f} | "
            f"AP {coco_stats['AP']:.4f} AP50 {coco_stats['AP50']:.4f} AP75 {coco_stats['AP75']:.4f} | "
            f"took {elapsed:.1f}s"
        )
    plot_metrics(
    os.path.join(log_dir, "metrics.csv"),
    os.path.join(log_dir, "plots"),
    smooth=0.2,
    style="dark",   # <= detta gör allt i %
    )
    # -------------------- EVAL --------------------
    #tries to load best ckpt path , fallback: best_no_aug
    try:
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE) 
    except:
        ckpt = torch.load(best_no_aug, map_location=DEVICE) 
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    evaluate_model(model=model, val_loader=val_loader, log_dir=log_dir, NUM_CLASSES=NUM_CLASSES, DEVICE=DEVICE, IMG_SIZE=IMG_SIZE, batch_size=batch_size)


    
    














