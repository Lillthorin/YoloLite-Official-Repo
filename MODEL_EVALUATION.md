# Evaluation & Plotting in YOLOLite

This document explains how **evaluation, metric computation, and plotting** work in the YOLOLite repository. The goal is to make results **transparent, reproducible, and interpretable**, especially when comparing CPU‑oriented models.

---

## 1. Overview of the Evaluation Pipeline

Evaluation is performed during or after training using a COCO‑style evaluation pipeline. Predictions are matched against ground truth annotations using **IoU thresholds**, and a wide range of metrics are computed.

Key characteristics:

* COCO‑compatible metrics (AP, AR)
* Class‑aware evaluation
* Confidence‑sweep analysis (Precision / Recall / F1 vs threshold)
* CSV‑based logging for reproducibility

---

## 2. Loss Curves (Training vs Validation)

During training, both **training loss** and **validation loss** are logged per epoch.

Typical interpretation:

* Rapid loss drop in early epochs → learning is stable
* Validation loss slightly below training loss is expected when augmentations are enabled
* Long, slow convergence indicates capacity‑limited or regularized models (common for CPU‑optimized networks)

![Loss Curve](images/loss_curve.png)

---

## 3. Confidence Sweep Analysis (IoU = 0.50)

Instead of reporting metrics at a single fixed confidence threshold, YOLOLite evaluates **Precision, Recall, and F1 across the full confidence range [0.0 – 1.0]**.

This provides:

* Dataset‑specific optimal confidence thresholds
* Better understanding of FP vs FN trade‑offs
* More realistic deployment guidance

### 3.1 Precision vs Confidence

Precision increases with higher confidence thresholds, as low‑confidence false positives are filtered out.

![Precision vs Confidence](images/P_curve.png)

---

### 3.2 Recall vs Confidence

Recall remains high at low and medium thresholds, then drops sharply when confidence becomes too strict.

![Recall vs Confidence](images/R_curve.png)

---

### 3.3 F1 Score vs Confidence

The F1 curve combines Precision and Recall. The **peak of this curve defines the best operating point** for deployment.

![F1 vs Confidence](images/F1_curve.png)

The confidence value at maximum F1 is automatically recorded as `best_conf`.

---

## 4. Metric Summary (COCO‑Style)

After evaluation, a full metric summary is generated.

Reported metrics include:

* **mAP@50** – detection quality at IoU 0.50
* **mAP@50:95** – stricter, averaged COCO metric
* **APS / APM / APL** – performance by object size
* **AR@50:95** – average recall
* **Best F1, Precision, Recall** – at optimal confidence
* **Inference latency (CPU / GPU)**

![Metric Summary](images/summary.png)

---

## 5. CSV Logging & Reproducibility

All evaluation results are written to a CSV file (`metrics.csv`). Each row corresponds to one epoch or evaluation run.

Typical columns:

```
epoch, AP, AP50, AP75, APS, APM, APL, AR,
train_loss, val_loss,
best_f1, best_conf,
precision, recall
```

Benefits:

* Easy plotting with pandas / matplotlib
* Long‑term experiment tracking
* Fair comparison across backbones and configs

---

## 6. Notes on Interpretation

* High **mAP@50** with lower **mAP@50:95** is common for datasets with loose boxes or annotation noise
* Validation vs test discrepancies (e.g. RF100) are often dataset‑driven, not model bugs
* CPU‑optimized models may trade localization sharpness for speed while retaining strong F1

---

## 7. Where Files Are Stored

Each training run produces a self‑contained directory:

```
runs/train/<id>/
 ├── metrics.csv
 ├── loss_curve.png
 ├── P_curve.png
 ├── R_curve.png
 ├── F1_curve.png
 ├── summary.png
 └── merged_config.yaml
```

This structure ensures every result can be reproduced from configuration + logs alone.

---

## 8. Metric Evolution Over Training (COCO Metrics)

In addition to final summary values, YOLOLite tracks **how each COCO metric evolves over training epochs**. This makes it easier to:

* Detect early saturation or instability
* Compare convergence speed across backbones
* Distinguish noise from real improvements

To reduce visual noise, an **Exponential Moving Average (EMA)** with (lpha = 0.20) is plotted alongside raw values.

---

### 8.1 AP (mAP@50:95)

AP reflects overall localization and classification quality across multiple IoU thresholds.

![AP](images/AP.png)

---

### 8.2 AP50

AP50 is less sensitive to box tightness and often converges earlier than AP@50:95.

![AP50](images/AP50.png)

---

### 8.3 AP75

AP75 emphasizes **precise localization** and is more sensitive to box quality.

![AP75](images/AP75.png)

---

### 8.4 APS / APM / APL (Object Size Breakdown)

YOLOLite reports performance separately for small, medium, and large objects.

**Small objects (APS):**
![APS](images/APS.png)

**Medium objects (APM):**
![APM](images/APM.png)

**Large objects (APL):**
![APL](images/APL.png)

---

### 8.5 AR (Average Recall)

Average Recall reflects the model’s ability to find objects independent of confidence ranking.

![AR](images/AR.png)

---

### 8.6 Combined COCO Metrics Overview

For a compact comparison, all major COCO metrics are plotted together.

![COCO Metrics Overview](images/metrics_overview.png)

This visualization highlights:

* Relative convergence speed of AP vs AR
* How small-object performance lags behind large objects
* Overall training stability

---

## 9. Confusion Matrix & Error Analysis

To complement aggregate metrics, YOLOLite generates a **class-wise confusion matrix** at the selected deployment threshold.

This example is evaluated at:

* **IoU = 0.50**
* **Confidence ≥ 0.74** (best F1 operating point)

![Confusion Matrix](images/confusion_matrix.png)

---

### 9.1 How to Read the Confusion Matrix

* **Rows** represent ground-truth classes
* **Columns** represent predicted classes
* Values are **row-normalized** (each row sums to 1.0)

Diagonal values correspond to correct classifications, while off-diagonal values represent misclassifications or background confusions.

---

### 9.2 Quantitative Error Breakdown

In addition to the visualization, YOLOLite logs explicit TP / FP / FN counts:

```
Total FP: 1
Total FN: 5

Class       TP   FP   FN   Precision   Recall
opluggad   38    0    1    1.000       0.974
pluggad    46    0    3    1.000       0.939
rulle      88    1    1    0.989       0.989
```

These values are derived from the same evaluation run and threshold.

---

### 9.3 Interpretation (Industrial Dataset)

This confusion matrix is typical for **well-annotated industrial datasets**:

* Near-perfect diagonals indicate low class ambiguity
* Errors are dominated by **missed detections (FN)** rather than false positives
* Minimal confusion between semantic classes

Important notes:

* High diagonal dominance **does not imply overfitting** when:

  * Validation and training data are separated
  * Metrics generalize across epochs
  * Recall remains stable across confidence thresholds

* In industrial inspection tasks, such matrices are expected due to:

  * Controlled imaging conditions
  * Clear object definitions
  * Limited class overlap

---

### 9.4 Background Class Behavior

The background column captures:

* Missed detections (objects predicted as background)
* Conservative behavior at higher confidence thresholds

This aligns with the confidence-sweep analysis, where recall drops sharply beyond the optimal threshold.

---

## 10. Running `evaluation.py` on a Test Dataset

In addition to validation-time evaluation during training, YOLOLite provides a standalone **`evaluation.py`** script that can be run on a dedicated **test folder**.

This mode is intended for:

* Final model assessment
* Dataset-independent benchmarking
* Reporting results without training-time bias

---

### 10.1 What `evaluation.py` Produces

When executed on a test set, `evaluation.py` generates the same artifacts as validation, but in a **single consolidated evaluation run**:

* COCO-style metric report (text)
* Precision / Recall / F1 vs confidence curves
* Confusion matrix at optimal confidence
* Per-class TP / FP / FN statistics
* Summary dashboard

This ensures consistency between validation and test evaluation.

---

### 10.2 COCO Metric Report (Test Set)

The script prints a full COCO-style summary:

```
AP          : 0.9141
AP50        : 0.9974
AP75        : 0.9939

APL         : 0.9638
APM         : 0.8941
APS         : 0.8652
AR          : 0.9381
ARL         : 0.9793
ARM         : 0.9286
ARS         : 0.8947
```

These values are computed on the **entire test set**, independent of training history.

---

### 10.3 Confidence Sweep (Test Set)

As with validation, metrics are evaluated across the full confidence range.

![F1 vs Confidence](images/F1_curve.png)
![Precision vs Confidence](images/P_curve.png)
![Recall vs Confidence](images/R_curve.png)

The optimal operating point is automatically selected based on maximum F1.

---

### 10.4 Metric Summary Dashboard

A compact summary is generated for quick inspection:

![Metric Summary](images/summary.png)

Reported values include:

* Best F1 score and threshold
* Precision / Recall at best F1
* mAP@50 and mAP@50:95
* Object-size breakdown (APS / APM / APL)
* CPU and GPU inference latency

---

### 10.5 Confusion Matrix & Error Counts (Test Set)

At the selected confidence threshold, a confusion matrix is produced:

![Confusion Matrix](images/confusion_matrix.png)

Alongside explicit error counts:

```
Total FP: 3
Total FN: 12

Class       TP   FP   FN   Precision   Recall
class0     129   0    9    1.000       0.935
class1     211   0    3    1.000       0.986
class2     352   3    0    0.992       1.000
```

Compared to validation results, test-set evaluation typically shows:

* Slightly lower recall due to harder samples
* Similar precision when class definitions are stable

---

### 10.6 Validation vs Test: Intended Usage

Recommended workflow:

* **Validation**: model selection, hyperparameter tuning
* **Test**: final reporting, cross-model comparison

