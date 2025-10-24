import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

def clip_boxes_and_labels(boxes, labels, w, h):
    clipped_boxes, clipped_labels = [], []
    for (x_min, y_min, x_max, y_max), lbl in zip(boxes, labels):
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w - 1))
        y_max = max(0, min(y_max, h - 1))
        if x_max > x_min and y_max > y_min:
            clipped_boxes.append([x_min, y_min, x_max, y_max])
            clipped_labels.append(lbl)
    return clipped_boxes, clipped_labels
def _parse_yolo_line(line):
    parts = line.strip().split()
    if len(parts) < 5:  # tom eller korrupt rad
        return None
    #Added support to handle segmented yolov8 datasets! 
    if len(parts) > 5:
        try:
            pts01 = np.asarray(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            xmin, ymin = pts01.min(axis=0)
            xmax, ymax = pts01.max(axis=0)
            cx = (xmin + xmax) * 0.5
            cy = (ymin + ymax) * 0.5
            w  = (xmax - xmin)
            h  = (ymax - ymin)
            # skydda mot degenererade polygoner
            eps = 1e-8
            return int(parts[0]), float(cx), float(cy), float(max(w, eps)), float(max(h, eps))
        except:
            return None
    cls, xc, yc, bw, bh = map(float, parts[:5])  # ignorera ev. conf/extra
    return int(cls), xc, yc, bw, bh


class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms, img_size=640, is_train=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_files = sorted([p for p in self.img_dir.glob("*.jpg")])
        self.label_files = [self.label_dir / (p.stem + ".txt") for p in self.img_files]

        self.img_size = img_size
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def load_image_and_labels(self, idx):
        img_path = str(self.img_files[idx])
        label_path = str(self.label_files[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parsed = _parse_yolo_line(line)
                    if parsed is None:
                        continue
                    cls, xc, yc, bw, bh = parsed
                    x_min = (xc - bw / 2) * w
                    y_min = (yc - bh / 2) * h
                    x_max = (xc + bw / 2) * w
                    y_max = (yc + bh / 2) * h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(cls)

        return img, boxes, labels

    # ------------------ Mosaic ------------------
    def mosaic(self, index):
        try:
            indices = [index] + random.choices(range(len(self.img_files)), k=3)
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

            mosaic_img = np.full((self.img_size * 2, self.img_size * 2, 3), 114, dtype=np.uint8)
            final_boxes, final_labels = [], []

            for i, idx in enumerate(indices):
                img_path = self.img_files[idx]
                label_path = self.label_files[idx]

                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]

                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                scale_x, scale_y = self.img_size / w, self.img_size / h

                offset_x = positions[i][1] * self.img_size
                offset_y = positions[i][0] * self.img_size

                mosaic_img[offset_y:offset_y + self.img_size,
                           offset_x:offset_x + self.img_size] = img_resized

                if os.path.exists(label_path):
                    with open(label_path) as f:
                        for line in f:
                            cls, xc, yc, bw, bh = map(float, line.strip().split())
                            cls = int(cls)
                            x1 = (xc - bw / 2) * w * scale_x + offset_x
                            y1 = (yc - bh / 2) * h * scale_y + offset_y
                            x2 = (xc + bw / 2) * w * scale_x + offset_x
                            y2 = (yc + bh / 2) * h * scale_y + offset_y
                            if x2 > x1 and y2 > y1:
                                final_boxes.append([x1, y1, x2, y2])
                                final_labels.append(cls)

            return mosaic_img, final_boxes, final_labels
        except:
            return self.load_image_and_labels(index)

    # ------------------ CutMix ------------------
    def cutmix(self, img, boxes, labels, other_idx, alpha=0.5):
        try:
            img2, boxes2, labels2 = self.load_image_and_labels(other_idx)
            h, w = img.shape[:2]

            cut_w = random.randint(int(w * 0.25), int(w * 0.5))
            cut_h = random.randint(int(h * 0.25), int(h * 0.5))
            cx, cy = random.randint(0, w - cut_w), random.randint(0, h - cut_h)

            img2 = cv2.resize(img2, (w, h))
            patch2 = img2[cy:cy+cut_h, cx:cx+cut_w]

            img[cy:cy+cut_h, cx:cx+cut_w] = (
                alpha * patch2 + (1 - alpha) * img[cy:cy+cut_h, cx:cx+cut_w]
            ).astype(np.uint8)

            new_boxes, new_labels = [], []
            for (x1, y1, x2, y2), lbl in zip(boxes2, labels2):
                if x2 < cx or x1 > cx+cut_w or y2 < cy or y1 > cy+cut_h:
                    continue
                nx1, ny1 = max(x1, cx), max(y1, cy)
                nx2, ny2 = min(x2, cx+cut_w), min(y2, cy+cut_h)
                if nx2 > nx1 and ny2 > ny1:
                    new_boxes.append([nx1, ny1, nx2, ny2])
                    new_labels.append(lbl)

            return img, boxes + new_boxes, labels + new_labels
        except:
            return img, boxes, labels

    # ------------------ CutMix fokus på små ------------------
    def cutmix_focus_small(self, img, boxes, labels, other_idx, alpha=0.7):
        """
        Klipper in den minsta boxen från en annan bild utan extra augmentering.
        Endast slutgiltig transform körs senare i __getitem__.
        """
        try:
            img2, boxes2, labels2 = self.load_image_and_labels(other_idx)
            h, w = img.shape[:2]

            if not boxes2:
                return img, boxes, labels

            # hitta minsta boxen
            areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes2]
            smallest_idx = int(np.argmin(areas))
            x1, y1, x2, y2 = boxes2[smallest_idx]

            # extrahera patch
            patch = img2[int(y1):int(y2), int(x1):int(x2)]
            if patch.size == 0:
                return img, boxes, labels

            ph, pw = patch.shape[:2]
            min_size = int(0.15 * min(w, h))

            # skala upp om patchen är för liten
            if ph < min_size or pw < min_size:
                scale = max(min_size / max(ph, 1), min_size / max(pw, 1))
                patch = cv2.resize(patch, (int(pw * scale), int(ph * scale)))
                ph, pw = patch.shape[:2]

            # säkerställ att patchen ryms i målbilden
            if pw >= w or ph >= h:
                return img, boxes, labels

            cx = random.randint(0, max(0, w - pw))
            cy = random.randint(0, max(0, h - ph))

            # klistra in med alpha-blend
            img[cy:cy+ph, cx:cx+pw] = (
                alpha * patch + (1 - alpha) * img[cy:cy+ph, cx:cx+pw]
            ).astype(np.uint8)

            # ny box
            new_box = [cx, cy, cx + pw, cy + ph]

            # returnera med nya boxen
            return img, boxes + [new_box], labels + [labels2[smallest_idx]]

        except Exception as e:
            print(f"[CutMix Fallback] {e}")
            return img, boxes, labels


    # ------------------ __getitem__ ------------------
    def __getitem__(self, idx):
        img, boxes, labels = self.load_image_and_labels(idx)

       
            
        if self.is_train:
            p = random.random()
            if p < 0.2:  # 20% Mosaic
                img, boxes, labels = self.mosaic(idx)
            elif p < 0.4:  # 20% CutMix fokus små
                img, boxes, labels = self.cutmix_focus_small(
                    img, boxes, labels, random.randint(0, len(self)-1)
                )
            else:  # 60% vanlig bild (ingen mosaic/cutmix)
                pass
            

        h, w = img.shape[:2]

        # KLIPP + SYNKA
        boxes, labels = clip_boxes_and_labels(boxes, labels, w, h)

        # sista skyddet om något ovan gett mismatch (t.ex. tomma listor från mosaic/cutmix)
        if len(boxes) != len(labels):
            n = min(len(boxes), len(labels))
            boxes, labels = boxes[:n], labels[:n]

        try:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
        except Exception as e:
            # Hjälper dig hitta filen som bråkar
            print(f"[DATASET ERROR] {self.img_files[idx]} len(b)={len(boxes)} len(l)={len(labels)} err={e}")
            raise

        img = transformed["image"]
        boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)


        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img, target


