import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms, img_size=640, is_train=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        
        # --- FIX: Använd os.scandir korrekt ---
        # Vi måste plocka ut .path från entryt, annars får vi <DirEntry object>
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.img_files = []
        
        # os.scandir är snabbt, men vi måste hantera det som en iterator
        # Om self.img_dir är ett Path-objekt, konvertera till sträng för os.scandir
        if os.path.exists(self.img_dir):
            with os.scandir(str(self.img_dir)) as entries:
                for entry in entries:
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts:
                        self.img_files.append(entry.path) # VIKTIGT: .path
        
        # Sortera för att garantera ordning (viktigt för determinism)
        self.img_files.sort()
        
        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

        self.img_size = img_size
        self.is_train = is_train
        self.transforms = transforms
        
        # --- CACHING ---
        print(f"Caching labels for {len(self.img_files)} images...")
        self.labels_cache = self._cache_labels()

    def _cache_labels(self):
        """Läser alla .txt filer och sparar i en lista av np.arrays."""
        cache = []
        # Använd tqdm för att se progress vid start
        for img_path in tqdm(self.img_files, desc="Loading labels"):
            label_path = self.label_dir / (Path(img_path).stem + ".txt")
            boxes = []
            if label_path.exists():
                try:
                    # Snabbare att läsa hela filen och splitta i minnet
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5: # Klassisk YOLO
                            # Hantera både vanlig YOLO och segmentering (förenklad)
                            cls = int(parts[0])
                            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                            
                            if len(coords) > 4: # Segmentering -> gör om till bbox
                                coords = coords.reshape(-1, 2)
                                xmin, ymin = coords.min(axis=0)
                                xmax, ymax = coords.max(axis=0)
                                xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
                                w, h = (xmax - xmin), (ymax - ymin)
                            else:
                                xc, yc, w, h = coords[:4]
                            
                            # Spara normalized [cls, xc, yc, w, h]
                            boxes.append([cls, xc, yc, w, h])
                except Exception:
                    pass
            
            # Spara som numpy array för snabbare indexering senare [N, 5]
            if boxes:
                cache.append(np.array(boxes, dtype=np.float32))
            else:
                cache.append(np.zeros((0, 5), dtype=np.float32))
        return cache

    def __len__(self):
        return len(self.img_files)

    def load_image(self, idx):
        """Läser enbart bilden (I/O tungt)."""
        img = cv2.imread(self.img_files[idx])
        # Hantera korrupta bilder
        if img is None:
            raise ValueError(f"Image not found or corrupt: {self.img_files[idx]}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_label_processed(self, idx, img_h, img_w):
        """Hämtar label från cache och konverterar till [x1, y1, x2, y2]."""
        # data är [N, 5] -> cls, xc, yc, w, h (normalized)
        data = self.labels_cache[idx]
        if data.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        cls = data[:, 0].astype(np.int64)
        xywh = data[:, 1:]

        # Vektoriserad konvertering xywh -> xyxy
        # x_min = (xc - w/2) * img_w
        x1 = (xywh[:, 0] - xywh[:, 2] / 2) * img_w
        y1 = (xywh[:, 1] - xywh[:, 3] / 2) * img_h
        x2 = (xywh[:, 0] + xywh[:, 2] / 2) * img_w
        y2 = (xywh[:, 1] + xywh[:, 3] / 2) * img_h
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        return boxes, cls

    def _clip_boxes(self, boxes, h, w):
        """NumPy optimerad clipping."""
        if len(boxes) == 0:
            return boxes
        # np.clip är extremt snabbt
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes

    # ------------------ Mosaic (Optimerad) ------------------
    def mosaic(self, index):
        indices = [index] + random.choices(range(len(self)), k=3)
        
        # Skapa canvas
        s = self.img_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        
        # Offsets: (y, x)
        offsets = [(0, 0), (0, s), (s, 0), (s, s)] 
        
        all_boxes, all_labels = [], []

        for i, idx in enumerate(indices):
            # Ladda bild
            img = self.load_image(idx)
            h, w = img.shape[:2]
            
            # Ladda label från RAM (instant)
            boxes, labels = self.load_label_processed(idx, h, w)

            # Resize bild
            r = s / max(h, w) # Behåll aspect ratio om du vill, eller kör hårt:
            if r != 1: 
                img = cv2.resize(img, (s, s))
                # Skala boxar
                scale_x = s / w
                scale_y = s / h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                h, w = s, s

            # Placera i mosaic
            oy, ox = offsets[i]
            mosaic_img[oy:oy+h, ox:ox+w] = img

            # Justera box-koordinater
            boxes[:, [0, 2]] += ox
            boxes[:, [1, 3]] += oy
            
            all_boxes.append(boxes)
            all_labels.append(labels)

        # Slå ihop allt med numpy
        if all_boxes:
            final_boxes = np.vstack(all_boxes)
            final_labels = np.concatenate(all_labels)
            
            # Filtrera bort ogiltiga boxar (x2 <= x1 osv)
            valid = (final_boxes[:, 2] > final_boxes[:, 0]) & (final_boxes[:, 3] > final_boxes[:, 1])
            return mosaic_img, final_boxes[valid], final_labels[valid]
        else:
            return mosaic_img, np.zeros((0,4)), np.zeros((0,))


    # ------------------ CutMix (Fokus små) ------------------
    def cutmix_focus_small(self, img, boxes, labels, other_idx, alpha=0.7):
        # Ladda andra bilden
        img2 = self.load_image(other_idx)
        h2, w2 = img2.shape[:2]
        boxes2, labels2 = self.load_label_processed(other_idx, h2, w2)

        if len(boxes2) == 0:
            return img, boxes, labels

        # Hitta minsta box (NumPy style)
        areas = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        smallest_idx = np.argmin(areas)
        
        # Extrahera data
        x1, y1, x2, y2 = boxes2[smallest_idx].astype(int)
        lbl = labels2[smallest_idx]
        
        patch = img2[y1:y2, x1:x2]
        if patch.size == 0: return img, boxes, labels
        
        ph, pw = patch.shape[:2]
        h, w = img.shape[:2]

        # --- FIX START: Kontrollera att patchen får plats ---
        if ph >= h or pw >= w:
            # Alternativ A: Patchen är större än målbilden -> Hoppa över cutmix denna gång
            return img, boxes, labels
            
            # Alternativ B (om du hellre vill det): Skala ner patchen
            # scale = min(h/ph, w/pw) * 0.9
            # patch = cv2.resize(patch, (int(pw*scale), int(ph*scale)))
            # ph, pw = patch.shape[:2]
        # --- FIX END ---

        # Enkel logik för paste
        cx = random.randint(0, max(0, w - pw))
        cy = random.randint(0, max(0, h - ph))
        
        # Klistra in
        roi = img[cy:cy+ph, cx:cx+pw]
        
        # --- SÄKERHETSCHECK: Om NumPy slicade ROI pga kanterna ---
        if roi.shape[:2] != patch.shape[:2]:
            return img, boxes, labels

        img[cy:cy+ph, cx:cx+pw] = (alpha * patch + (1 - alpha) * roi).astype(np.uint8)

        # Lägg till ny box till existerande (NumPy concat)
        new_box = np.array([[cx, cy, cx+pw, cy+ph]], dtype=np.float32)
        new_lbl = np.array([lbl], dtype=np.int64)
        
        return img, np.vstack([boxes, new_box]), np.concatenate([labels, new_lbl])

    def __getitem__(self, idx):
        # Default: ladda en bild
        img = self.load_image(idx)
        h, w = img.shape[:2]
        boxes, labels = self.load_label_processed(idx, h, w)

        if self.is_train:
            p = random.random()
            if p < 0.2:
                img, boxes, labels = self.mosaic(idx)
            elif p < 0.4:
                # CutMix förutsätter att vi har en basbild, så vi använder den vi laddade
                img, boxes, labels = self.cutmix_focus_small(img, boxes, labels, random.randint(0, len(self)-1))
            
            # Klipp boxar så de håller sig inom bildens ramar (viktigt efter mosaic/cutmix)
            h, w = img.shape[:2]
            boxes = self._clip_boxes(boxes, h, w)

        try:
            # Albumentations vill ofta ha listor, men klarar numpy. 
            # Vi konverterar till listor enbart för anropet om det krävs, annars kör vi numpy.
            if len(boxes) > 0:
                transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed["image"]
                boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)
            else:
                # Hantera tomma fall (albumentations kraschar ibland på tomma listor)
                transformed = self.transforms(image=img, bboxes=[], class_labels=[])
                img = transformed["image"]
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
            return img, target

        except Exception as e:
            print(f"[ERROR] {self.img_files[idx]}: {e}")
            # Returnera en tom tensor istället för crash, så DataLoader kan fortsätta

            return torch.zeros((3, self.img_size, self.img_size)), {"boxes": torch.zeros((0,4)), "labels": torch.zeros((0,))}
