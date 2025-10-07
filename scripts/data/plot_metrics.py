#!/usr/bin/env python3
# tools/plot_metrics.py
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

METRIC_KEYS = ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR"]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def read_metrics_csv(csv_path: str) -> Dict[str, List[float]]:
    """
    Läser metrics.csv som skrivs av train.py (epoch, AP, AP50, AP75, APS, APM, APL, AR, ...).
    Returnerar en dict: key -> lista (index 0 = epoch 1).
    """
    data: Dict[str, List[float]] = {}
    csv_path = str(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Hittar inte CSV: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise RuntimeError("Tom header i CSV")

        # initiera listor för alla kolumner
        for h in header:
            data[h] = []

        for row in reader:
            if not row or len(row) != len(header):
                # hoppa över korrupt rad
                continue
            for h, v in zip(header, row):
                if h.lower() == "timestamp":
                    # spara tidsstämplar som strängar
                    data[h].append(v)
                else:
                    data[h].append(_safe_float(v))

    return data


def ema_smooth(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Enkel EMA-utjämning. alpha in [0..1], 0 = av, 1 = ingen smoothing (identitet).
    """
    if not len(x) or alpha <= 0.0:
        return x
    if alpha >= 1.0:
        return x
    y = np.empty_like(x, dtype=np.float64)
    m = 0.0
    for i, v in enumerate(x):
        if i == 0 or np.isnan(m):
            m = v
        else:
            m = alpha * v + (1.0 - alpha) * m
        y[i] = m
    return y


def _best_idx(values: np.ndarray) -> int:
    """Returnera index för max-värdet (ignorerar NaN)."""
    if values.size == 0:
        return -1
    if np.all(np.isnan(values)):
        return -1
    return int(np.nanargmax(values))


def plot_single_metric(
    epochs: np.ndarray,
    values: np.ndarray,
    out_dir: Path,
    name: str,
    smooth_alpha: float,
    style: str,
    as_percent: bool = True
):
    plt.style.use("default" if style == "light" else "seaborn-v0_8-dark")
    fig = plt.figure(figsize=(8, 5), dpi=140)
    ax = fig.add_subplot(111)

    y = values.astype(np.float64)
    y_s = ema_smooth(y, alpha=smooth_alpha) if smooth_alpha > 0 else y

    ax.plot(epochs, y, linewidth=1.2, alpha=0.35, label=f"{name} (raw)")
    ax.plot(epochs, y_s, linewidth=2.0, label=f"{name} (EMA {smooth_alpha:.2f})" if smooth_alpha > 0 else name)

    # markera bästa punkt (på smoothed kurva om smoothing används)
    best = _best_idx(y_s)
    if best >= 0:
        ax.scatter([epochs[best]], [y_s[best]], s=30, zorder=3)
        ax.annotate(
            f"best={y_s[best]:.4f} @ ep {int(epochs[best])}",
            (epochs[best], y_s[best]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


def plot_overview(
    epochs: np.ndarray,
    series: Dict[str, np.ndarray],
    out_dir: Path,
    smooth_alpha: float,
    style: str,
):
    plt.style.use("default" if style == "light" else "seaborn-v0_8-dark")
    fig = plt.figure(figsize=(11, 6), dpi=140)
    ax = fig.add_subplot(111)

    for k in METRIC_KEYS:
        if k in series and series[k].size:
            y = series[k].astype(np.float64)
            y_s = ema_smooth(y, alpha=smooth_alpha) if smooth_alpha > 0 else y
            ax.plot(epochs, y_s, linewidth=2.0, label=k)

            # markera bästa
            best = _best_idx(y_s)
            if best >= 0:
                ax.scatter([epochs[best]], [y_s[best]], s=18, zorder=3)

    ax.set_title("COCO-metrics overview")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=4, fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_overview.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Sökväg till metrics.csv i din run-mapp")
    ap.add_argument("--out", required=True, help="Mapp att spara grafer i (t.ex. runs/train/1/plots)")
    ap.add_argument("--smooth", type=float, default=0.2, help="EMA-alpha (0 = av)")
    ap.add_argument("--style", choices=["light", "dark"], default="dark", help="Plot-tema")
    args = ap.parse_args()

    data = read_metrics_csv(args.csv)

    # plocka ut epochs
    if "epoch" not in data or not len(data["epoch"]):
        raise RuntimeError("CSV saknar kolumnen 'epoch' eller har inga rader.")
    epochs = np.array([int(v) if not math.isnan(v) else math.nan for v in data["epoch"]], dtype=np.float64)

    # serier för alla begärda metrics (tål saknade kolumner)
    series: Dict[str, np.ndarray] = {}
    for k in METRIC_KEYS:
        if k in data:
            series[k] = np.array([_safe_float(v) for v in data[k]], dtype=np.float64)
        else:
            series[k] = np.array([], dtype=np.float64)
            print(f"[Varning] Kolumn saknas i CSV: {k}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individuella grafer
    for k, vals in series.items():
        if vals.size:
            plot_single_metric(epochs, vals, out_dir, k, args.smooth, args.style)

    # Översiktsgraf
    plot_overview(epochs, series, out_dir, args.smooth, args.style)
    '''
    # (valfritt) plottar loss & LR om de finns
    if "train_loss" in data and "val_loss" in data:
        plt.style.use("default" if args.style == "light" else "seaborn-v0_8-dark")
        fig = plt.figure(figsize=(8, 5), dpi=140)
        ax = fig.add_subplot(111)
        tr = np.array([_safe_float(v) for v in data["train_loss"]], dtype=np.float64)
        vl = np.array([_safe_float(v) for v in data["val_loss"]], dtype=np.float64)
        ax.plot(epochs, ema_smooth(tr, args.smooth), label="train_loss", linewidth=2.0)
        ax.plot(epochs, ema_smooth(vl, args.smooth), label="val_loss", linewidth=2.0)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.25); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "loss.png")
        fig.savefig(out_dir / "loss.svg")
        plt.close(fig)

    # LRs (om tillgängliga)
    lr_keys = [k for k in data.keys() if k.lower().startswith("lr_")]
    if lr_keys:
        plt.style.use("default" if args.style == "light" else "seaborn-v0_8-dark")
        fig = plt.figure(figsize=(8, 5), dpi=140)
        ax = fig.add_subplot(111)
        for lk in lr_keys:
            y = np.array([_safe_float(v) for v in data[lk]], dtype=np.float64)
            ax.plot(epochs, y, label=lk, linewidth=2.0)
        ax.set_title("Learning rates")
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.grid(True, alpha=0.25); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "lrs.png")
        fig.savefig(out_dir / "lrs.svg")
        plt.close(fig)
    '''
    print(f"✅ Done! Plots saved in: {out_dir}")


# ===== Importvänlig funktion för anrop från train =====
def plot_metrics(csv_path: str, out_dir: str, smooth: float = 0.2, style: str = "dark"):
    """
    Exempel i din train-loop:
        from tools.plot_metrics import plot_metrics
        plot_metrics(os.path.join(log_dir, 'metrics.csv'), os.path.join(log_dir, 'plots'), smooth=0.2)
    """
    ns = argparse.Namespace(csv=csv_path, out=out_dir, smooth=smooth, style=style)
    # Mappa om till main-komponenterna
    data = read_metrics_csv(ns.csv)
    epochs = np.array([int(v) if not math.isnan(v) else math.nan for v in data["epoch"]], dtype=np.float64)
    series: Dict[str, np.ndarray] = {k: (np.array([_safe_float(v) for v in data[k]], dtype=np.float64)
                                        if k in data else np.array([], dtype=np.float64))
                                     for k in METRIC_KEYS}
    out_path = Path(ns.out); out_path.mkdir(parents=True, exist_ok=True)
    for k, vals in series.items():
        if vals.size:
            plot_single_metric(epochs, vals, out_path, k, ns.smooth, ns.style)
    plot_overview(epochs, series, out_path, ns.smooth, ns.style)
    
    print(f"✅ Done! Grafs saved in: {out_path}")


if __name__ == "__main__":
    main()
