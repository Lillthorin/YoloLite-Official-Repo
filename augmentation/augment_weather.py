import argparse, json, shutil, subprocess, random, os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from tqdm import tqdm

import albumentations as A

# -----------------------------
# Albumentations weather pipeline
# -----------------------------
def build_weather_pipeline(effects: List[str], p_each: float):
    # Finns: RandomRain, RandomSnow, RandomFog, RandomSunFlare, RandomShadow :contentReference[oaicite:2]{index=2}
    ops = []
    for e in effects:
        e = e.lower()
        if e == "rain":
            # Välj en slumpad regntyp varje gång transformen appliceras
            rain_types = ["drizzle", "heavy", "torrential"]

            rain_ops = [
                A.RandomRain(
                    brightness_coefficient=0.9,
                    drop_width=1,
                    blur_value=3,
                    rain_type=rt,
                    p=1.0,           # alltid aktiv inom denna OneOf
                )
                for rt in rain_types
            ]

            # (Valfritt) lägg till en "default"-variant utan explicit rain_type
            # om du vill behålla Albumentations standardbeteende ibland.
            rain_ops.append(
                A.RandomRain(
                    brightness_coefficient=0.9,
                    drop_width=1,
                    blur_value=3,
                    p=1.0
                )
            )

            # p=p_each styr sannolikheten att någon regnvariant används alls.
            # Inuti väljs exakt en av rain_ops med lika sannolikhet.
            ops.append(A.OneOf(rain_ops, p=p_each))
        elif e == "snow":
            ops.append(A.RandomSnow(brightness_coeff=2.0, snow_point_range=(0.2, 0.5), p=p_each))
        elif e == "fog":
            ops.append(A.RandomFog(fog_coef_range=(0.3, 0.6), alpha_coef=0.08, p=p_each))
        elif e == "sunflare":
            ops.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 0.7), angle_lower=0.5, p=p_each))
        elif e == "shadow":
            ops.append(A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, p=p_each))
        else:
            raise ValueError(f"Unknown effect: {e}")

    # Vi gör en "en-av-flera" vädereffekt per bild (kan ökas med Compose om du vill stapla dem).
    weather = A.OneOf(ops, p=1.0)

    # Inga geometriska operationer -> bboxar ändras ej, men vi skickar med params så formatet hanteras korrekt.
    transform = A.Compose(
        [weather],
        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], clip=True)
    )
    return transform

# -----------------------------
# Hjälpare för YOLO-format
# -----------------------------
def read_yolo_bboxes(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if len(parts) > 5:
                pts01 = np.asarray(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                xmin, ymin = pts01.min(axis=0)
                xmax, ymax = pts01.max(axis=0)
                cx = (xmin + xmax) * 0.5
                cy = (ymin + ymax) * 0.5
                w  = (xmax - xmin)
                h  = (ymax - ymin)
                # skydda mot degenererade polygoner
                eps = 1e-8
                cls = int(parts[0])
                boxes.append([float(cx), float(cy), float(max(w, eps)), float(max(h, eps))])
                labels.append(cls)
            else:  
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append([x, y, w, h])
                labels.append(cls)
    return boxes, labels

def write_yolo_bboxes(label_path: Path, boxes: List[List[float]], labels: List[int]):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for box, cls in zip(boxes, labels):
            x, y, w, h = box
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# -----------------------------
# Hjälpare för COCO-format
# -----------------------------
def load_coco(ann_path: Path) -> Dict[str, Any]:
    with open(ann_path, "r") as f:
        return json.load(f)

def save_coco(ann_path: Path, coco: Dict[str, Any]):
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump(coco, f)

def coco_add_augmented_image(coco: Dict[str, Any], img_rec: Dict[str, Any], anns: List[Dict[str, Any]], new_file_name: str, new_width: int, new_height: int) -> None:
    new_img_id = max([im["id"] for im in coco["images"]]) + 1 if coco["images"] else 1
    coco["images"].append({
        "id": new_img_id,
        "file_name": new_file_name,
        "width": new_width,
        "height": new_height
    })
    start_ann_id = max([a["id"] for a in coco["annotations"]]) + 1 if coco["annotations"] else 1
    next_id = start_ann_id
    for a in anns:
        # bbox i COCO är [x,y,w,h] i pixlar – vädereffekter ändrar inte geometri → oförändrad bbox/area
        new_ann = dict(a)
        new_ann["id"] = next_id
        new_ann["image_id"] = new_img_id
        coco["annotations"].append(new_ann)
        next_id += 1

# -----------------------------
# (Valfritt) rain-rendering hook
# Kräver djupkartor + struktur enligt deras README. :contentReference[oaicite:3]{index=3}
# -----------------------------
def run_rain_rendering(repo_dir: Path, dataset_name: str, intensity_mm: int = 25, frame_end: int = 0):
    cmd = [
        "python", str(repo_dir / "main.py"),
        "--dataset", dataset_name,
        "--intensity", str(intensity_mm)
    ]
    if frame_end > 0:
        cmd += ["--frame_end", str(frame_end)]
    subprocess.run(cmd, check=True)

# -----------------------------
# Format-detektion
# -----------------------------
def guess_format(root: Path) -> str:
    yolo = (root / "train" / "images").exists() and (root / "train" / "labels").exists()
    coco1 = (root / "train" / "annotations.json").exists()
    coco2 = (root / "annotations" / "train.json").exists()
    if yolo:
        return "yolo"
    if coco1 or coco2:
        return "coco"
    raise RuntimeError("Kunde inte hitta YOLO- eller COCO-struktur.")

# -----------------------------
# YOLO augmentering
# -----------------------------
def augment_yolo(root: Path, num_aug: int, effects: List[str], suffix: str, p_each: float):
    img_dir = root / "train" / "images"
    lbl_dir = root / "train" / "labels"
    transform = build_weather_pipeline(effects, p_each)

    images = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])

    for img_path in tqdm(images, desc="Augment YOLO(train)"):
        rel = img_path.relative_to(img_dir)
        base = rel.stem
        label_path = lbl_dir / rel.with_suffix(".txt")

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        boxes, labels = read_yolo_bboxes(label_path)
        if len(boxes) == 0:  # skip tomma annoteringar – kopiera ändå om du vill
            continue

        for k in range(num_aug):
            aug = transform(image=image, bboxes=boxes, labels=labels)
            out_img = aug["image"]
            out_boxes = aug["bboxes"]
            out_labels = aug["labels"]

            out_img_path = img_path.with_name(f"{base}{suffix}{k+1}{img_path.suffix}")
            out_lbl_path = label_path.with_name(f"{base}{suffix}{k+1}.txt")

            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), out_img)
            write_yolo_bboxes(out_lbl_path, out_boxes, out_labels)

# -----------------------------
# COCO augmentering
# -----------------------------
def augment_coco(root: Path, num_aug: int, effects: List[str], suffix: str, p_each: float):
    ann_path = root / "train" / "annotations.json"
    if not ann_path.exists():
        ann_path = root / "annotations" / "train.json"
    coco = load_coco(ann_path)

    # Indexera annotationer per bild
    anns_by_img = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)

    transform = build_weather_pipeline(effects, p_each)

    img_dir = root / "train" / "images"
    if not img_dir.exists():
        # ibland ligger filerna utan extra /images
        img_dir = root / "train"

    id_to_img = {im["id"]: im for im in coco.get("images", [])}
    for img in tqdm(coco.get("images", []), desc="Augment COCO(train)"):
        img_path = img_dir / img["file_name"]
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        # Hämta bboxar i YOLO-normaliserat format tillfälligt, så Compose kan hantera dem enhetligt
        w, h = img["width"], img["height"]
        anns = anns_by_img.get(img["id"], [])
        yolo_boxes, yolo_labels, keep_anns = [], [], []
        for a in anns:
            x, y, bw, bh = a["bbox"]  # COCO xywh pixlar
            if bw <= 0 or bh <= 0:
                continue
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            yolo_boxes.append([cx, cy, nw, nh])
            yolo_labels.append(int(a.get("category_id", 0)))
            keep_anns.append(a)

        if len(yolo_boxes) == 0:
            continue

        for k in range(num_aug):
            aug = transform(image=image, bboxes=yolo_boxes, labels=yolo_labels)
            out_img = aug["image"]

            # Spara bild
            stem = Path(img["file_name"]).stem
            ext = Path(img["file_name"]).suffix
            new_name = f"{stem}{suffix}{k+1}{ext}"
            out_path = img_dir / new_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), out_img)

            # Lägg in nya image + annotations (bbox återställs till COCO pixlar)
            new_w, new_h = out_img.shape[1], out_img.shape[0]
            # (Vi håller samma dimensioner — om du vill resize:a måste bboxarna återskalas)
            # Här antar vi att transformen inte ändrar storlek; Albumentations väder gör inte det per default. :contentReference[oaicite:4]{index=4}
            new_anns = []
            for a in keep_anns:
                new_a = dict(a)
                # bbox oförändrad i pixlar
                new_anns.append(new_a)

            coco_add_augmented_image(coco, img, new_anns, new_name, w, h)

    save_coco(ann_path, coco)

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Weather augmentation for detection datasets")
    ap.add_argument("--data", type=str, required=True, help="Path to dataset root (contains train/ ...)")
    ap.add_argument("--fmt", type=str, default="auto", choices=["auto","yolo","coco"], help="Dataset format")
    ap.add_argument("--effects", type=str, default="rain,snow,fog,sunflare,shadow", help="Comma-separated effects")
    ap.add_argument("--num_aug", type=int, default=1, help="Augmented copies per original")
    ap.add_argument("--suffix", type=str, default="_wx", help="Filename suffix for augmented images")
    ap.add_argument("--p_each", type=float, default=1.0, help="Probability per chosen weather op inside OneOf (kept at 1.0)")
    args = ap.parse_args()

    root = Path(args.data)
    fmt = args.fmt if args.fmt != "auto" else guess_format(root)
    effects = [e.strip() for e in args.effects.split(",") if e.strip()]


    if fmt == "yolo":
        augment_yolo(root, args.num_aug, effects, args.suffix, args.p_each)
    elif fmt == "coco":
        augment_coco(root, args.num_aug, effects, args.suffix, args.p_each)
    else:
        raise RuntimeError("Unknown format")

if __name__ == "__main__":
    main()

