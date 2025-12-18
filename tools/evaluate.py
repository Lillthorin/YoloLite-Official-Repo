# train.py
import os, sys
from pathlib import Path
import torch


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from tools.infer import load_model_names_imgsize_from_ckpt
from torch.utils.data import DataLoader

from scripts.data.dataset import YoloDataset
from scripts.data.augment import get_val_transform
from scripts.helpers.helpers import yolo_collate
from scripts.helpers.evaluate import evaluate_model
import argparse

def _next_run_dir(base: str) -> str:
    """
    Skapa och returnera nästa lediga run-mapp som en numerisk subdir under 'base'.
    Ex: base='runs' -> 'runs/1', 'runs/2', ...
        base='runs/weeds' -> 'runs/weeds/1', ...
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path tol checkpoint (.pt/.pth)")
    ap.add_argument("--test_folder", required=True, help="path to test/val folder")
    ap.add_argument("--img_size", type=int, default=0, help="Override meta.img_size)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no_letterbox", action="store_true", help="Use pure resize instead of letterbox")
    args = ap.parse_args()
    
    device = f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    
    model, names, meta_img_size = load_model_names_imgsize_from_ckpt(args.weights, device)
    test_images =  os.path.join(args.test_folder, 'images')
    test_labels =  os.path.join(args.test_folder, 'labels')
    log_dir_parent = os.path.join("runs/evaluate")
    os.makedirs(log_dir_parent, exist_ok=True)
    log_dir = _next_run_dir(log_dir_parent)
    
    val_resize = 1.0 if args.no_letterbox == True else 0.0
    test_ds = YoloDataset(
            test_images,
            test_labels,
            img_size=meta_img_size,
            is_train=False,
            transforms=get_val_transform(meta_img_size, val_resize)
        )
    test_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=yolo_collate,        
                drop_last=False,
            )
    
    evaluate_model(model=model, val_loader=test_loader, log_dir=log_dir, NUM_CLASSES=len(names), DEVICE=device, IMG_SIZE=meta_img_size, batch_size=args.batch_size, class_names=names)
    
    

if __name__ == "__main__":
    main()
