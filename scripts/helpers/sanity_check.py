import os, sys

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from scripts.data.dataset import YoloDataset
from scripts.data.augment import get_base_transform, get_val_transform
import matplotlib.patches as patches

def denormalize(img_tensor):
    """Convert tensor [C,H,W] back to numpy [H,W,C] (0-255)."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
    img = img.clip(0, 255).astype("uint8")
    return img

def visualize_batch(images, targets, save_path="sanity_check.jpg", max_images=4):
    bs = min(len(images), max_images)
    cols = int(np.ceil(np.sqrt(bs)))
    rows = int(np.ceil(bs / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for i, ax in enumerate(axes.flat):
        if i >= bs:
            ax.axis("off")
            continue

        img = denormalize(images[i])
        ax.imshow(img)

        boxes = targets[i]["boxes"].cpu().numpy()  # redan i x1,y1,x2,y2
        labels = targets[i]["labels"].cpu().numpy()

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h,
                                     linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            text = str(label)
            ax.text(x1, max(0, y1 - 2), text, color="yellow",
                    fontsize=10, weight="bold")

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Sanity check sparad till {save_path}")

if __name__ == "__main__":
    img_dir = r"C:\Users\skhMATN\OneDrive - Stora Enso OYJ\Skrivbordet\Test_modell\dataset\valid\images"
    label_dir = r"C:\Users\skhMATN\OneDrive - Stora Enso OYJ\Skrivbordet\Test_modell\dataset\valid\labels"
    img_size = 640
    batch_size = 8

    dataset = YoloDataset(
        img_dir, label_dir,
        transforms=get_base_transform(img_size),
        img_size=img_size,
        is_train=True
    )

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    images, targets = next(iter(loader))
    visualize_batch(images, targets, save_path="sanity_check.jpg")  # byt till dina klasser
