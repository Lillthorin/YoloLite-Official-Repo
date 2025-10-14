import argparse
import os
from pathlib import Path
from typing import Dict, Any
import yaml
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

def _update_latest_pointer(parent: str, target: str):
    """
    Försök skapa/uppdatera en 'latest' som pekar på target.
    - Unix: symbolisk länk 'latest' -> target
    - Windows utan admin: skriv en textfil 'latest.txt' med sökvägen
    """
    parent_p = Path(parent)
    latest_link = parent_p / "latest"

    try:
        # Radera existerande länk/fil
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        # Skapa symlink (kan kräva admin på Windows)
        latest_link.symlink_to(Path(target), target_is_directory=True)
    except Exception:
        # Fallback: skriv latest.txt
        try:
            with open(parent_p / "latest.txt", "w", encoding="utf-8") as f:
                f.write(str(Path(target)))
        except Exception:
            pass  # om även detta fallerar, hoppa tyst

def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _norm(p: str) -> str:
    # Normalisera till absolut sökväg och behåll Windows-stigar
    return str(Path(p).expanduser().resolve()) if p else p

def _infer_labels_dir(images_dir: str) -> str:
    # Byt .../images -> .../labels (Ultralytics-konvention)
    # Fungerar även om det heter "images" med olika case men vi håller det enkelt
    if images_dir is None:
        return None
    parts = Path(images_dir).parts
    if len(parts) >= 1 and parts[-1].lower() == "images":
        return str(Path(*parts[:-1], "labels"))
    # Om ingen "images" – anta syskonmapp "labels" bredvid
    return str(Path(images_dir).parent / "labels")

def _ensure_dir_exists(p: str, tag: str):
    if p and not Path(p).exists():
        raise FileNotFoundError(f"{tag} path not found: {p}")

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge b into a (recursively), returning a."""
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def load_configs(model_yaml: str, train_yaml: str, data_yaml: str) -> Dict[str, Any]:
    model_yaml = _norm(model_yaml) if model_yaml else None
    train_yaml = _norm(train_yaml) if train_yaml else None
    data_yaml  = _norm(data_yaml)  if data_yaml else None

    model_cfg = _read_yaml(model_yaml) if model_yaml else {}
    train_cfg = _read_yaml(train_yaml) if train_yaml else {}
    data_cfg  = _read_yaml(data_yaml)  if data_yaml  else {}

    # --- Bygg intern "dataset"-sektion från data.yaml ---
    data_train_images = _norm(data_cfg.get("train", ""))
    data_val_images   = _norm(data_cfg.get("val", ""))
    data_test_images  = _norm(data_cfg.get("test", ""))
    # Härled labels-mappar:
    train_labels = _infer_labels_dir(data_train_images) if data_train_images else ""
    val_labels   = _infer_labels_dir(data_val_images)   if data_val_images   else ""
    test_labels  = _infer_labels_dir(data_test_images)  if data_test_images  else ""

    # Validera att bilder/labels finns (om satta)
    for tag, p in [("train_images", data_train_images),
                   ("val_images", data_val_images),
                   ("test_images", data_test_images)]:
        if p:
            _ensure_dir_exists(p, tag)

    for tag, p in [("train_labels", train_labels),
                   ("val_labels", val_labels),
                   ("test_labels", test_labels)]:
        if p:
            _ensure_dir_exists(p, tag)

    names = data_cfg.get("names")
    if names is not None and not isinstance(names, (list, tuple)):
        raise ValueError("data.yaml 'names' must be a list of class names.")
    nc = data_cfg.get("nc", len(names) if names else None)
    if nc is None:
        raise ValueError("Unable to infer 'nc'. Please set 'nc' or provide 'names' in data.yaml.")

    dataset_block = {
        "dataset": {
            # Håller kvar tidigare fält (om din kod förväntar dem):
            "train_images": data_train_images,
            "val_images": data_val_images,
            "test_images": data_test_images,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "test_labels": test_labels,
            "names": list(names) if names else [str(i) for i in range(nc)],
        }
    }

    # --- Sätt num_classes i model om ej explicit ---
    model_block = model_cfg.get("model", {})
    if "num_classes" not in model_block or model_block.get("num_classes") is None:
        model_block["num_classes"] = int(nc)
    model_cfg["model"] = model_block

    # --- Standardvärden om de saknas ---
    # img_size kan bo i training.img_size; fallback till 640
    if "training" not in train_cfg:
        train_cfg["training"] = {}
    if "img_size" not in train_cfg["training"]:
        # Behåll kompatibilitet med gamla "dataset.img_size"
        ds_img_size = model_cfg.get("dataset", {}).get("img_size") or train_cfg.get("dataset", {}).get("img_size")
        train_cfg["training"]["img_size"] = int(ds_img_size) if ds_img_size else 640

    # --- Slå ihop:  dataset_block + model_cfg + train_cfg -> config ---
    config: Dict[str, Any] = {}
    _deep_merge(config, dataset_block)
    _deep_merge(config, model_cfg)
    _deep_merge(config, train_cfg)

    # --- Säkerställ logging + auto-inkrementerande run-dir ---
    log_cfg = config.get("logging", {})
    base_log_dir = log_cfg.get("log_dir")  # kan vara None eller t.ex. 'runs' / 'runs/weeds'

    # Standardbas om inget angetts
    if not base_log_dir or not str(base_log_dir).strip():
        base_log_dir = "runs"

    # Skapa nästa run-katalog under basen
    run_dir = _next_run_dir(base_log_dir)
    config["logging"] = {"log_dir": run_dir}

    # (Valfritt) uppdatera en 'latest' pekare i basmappen
    try:
        _update_latest_pointer(parent=str(Path(run_dir).parent), target=run_dir)
    except Exception:
        pass


    return config

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="configs/models/yololite_m.yaml", type=str, required=True, help="Path to model.yaml")
    ap.add_argument("--train", default="configs/train/standard_train.yaml", type=str, required=False, help="Path to train.yaml")
    ap.add_argument("--data",   type=str, required=True, help="Path to data.yaml")
    # Vanliga CLI overrides (valfria)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=0)   # t.ex. "0", "cpu"
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--augment", type=bool, default=True)
    ap.add_argument("--use_p6", type=bool, default=False)
    ap.add_argument("--resume", type=str, default=None, help="Resume training from last checkpoint if available")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate if set")
    ap.add_argument("--save_every", type=int, default=25, help="Save every x epoch")
    ap.add_argument("--save_by", type=str, default='AP', help="Save best model by coco evaluation, viable setting [AP50, AP75, AP, AR, APS, APM, APL]")  
    
    return ap

def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # CLI ska få sista ordet
    ok_save_by = ["AP50", "AP75", "AP", "AR", "APS", "APM", "APL"]
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.img_size is not None:
        config["training"]["img_size"] = int(args.img_size)
    if args.workers is not None:
        config["training"]["num_workers"] = int(args.workers)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.use_p6 is not None:
        config["training"]["use_p6"] = bool(args.use_p6)
    if args.augment is not None:
        config["training"]["augment"] = bool(args.augment)
    if args.resume is not None:
        config["training"]["resume"] = str(args.resume)
    if args.lr is not None:
        config["training"]["lr"] = float(args.lr)
    if args.save_every is not None:
        config["training"]["save_every"] = int(args.save_every)
    if args.save_by is not None:
        if args.save_by in ok_save_by:
            config["training"]["save_by"] = str(args.save_by)
        else:
            print("Invalid token for save_by. Valid tokens: [AP50, AP75, AP, AR, APS, APM, APL,P, R, F1]")
            raise ValueError
    return config
