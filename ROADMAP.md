# What's Next – Project Roadmap

This document outlines the planned next steps for the project.  
The roadmap is intentionally pragmatic and reflects both technical priorities and real-world constraints such as time, hardware access, and budget.

The main focus going forward remains **edge-friendly, deterministic object detection**, with special attention to **CPU and low-power GPU environments**.

---

## Step 1 – Expand the *Edge* Series with GPU-Oriented Variants
The first priority is to extend the existing **Edge model series** with variants optimized for **weaker GPUs**, such as:

- Jetson Nano / Jetson Orin Nano
- Low-power embedded GPUs
- Industrial PCs with limited GPU resources

These models will retain the core design philosophy of the Edge series:
- Low parameter count
- Predictable performance
- Configurable backbone / neck / head
- Focus on real-time inference under constrained hardware

The goal is **not** high-end GPU benchmarking, but practical deployment on modest GPU hardware.

---

## Step 2 – Backbone Selection & Usage Guide
A comprehensive **backbone guide** will be created based on large-scale testing.

This guide will:
- Compare a wide range of backbones (via `timm`)
- Show performance trade-offs between accuracy, speed, and model size
- Help users understand **when custom backbones make sense**
- Provide recommendations for:
  - CPU-only
  - Weak GPU
  - Balanced edge deployments

The intention is to give users confidence when deviating from default configurations.

---

## Step 3 – Benchmarks on Real Hardware
Current benchmarks are largely system-based.  
A future step is to run **controlled benchmarks on actual edge hardware**, such as:

- Raspberry Pi variants
- Jetson devices
- Industrial edge PCs

This step is **budget-dependent** and will be performed once suitable hardware can be acquired.  
The focus will be on **reproducibility and transparency**, not synthetic peak numbers.

---

## Step 4 – Exponential Moving Average (EMA) During Training
EMA support will be integrated into the training and validation pipeline.

Expected benefits:
- More stable convergence
- Potential accuracy improvements
- Better generalization, especially on smaller datasets

EMA will be optional and configurable to avoid unnecessary overhead where not needed.

---

## Step 5 – Iterative Improvements to Augmentation, Neck, and Loss
Augmentations, neck design, and loss formulation will be **continuously reviewed and refined**.

This includes:
- Dataset-specific augmentation tuning
- Neck structure experiments
- Loss balancing and assignment strategies

This step is ongoing by nature and will progress as time allows, rather than being tied to a strict milestone.

---

## Step 6 – Expanded Deployment & Export Options
Deployment is a core goal of the project.

Planned improvements include:
- Additional export formats
- Improved ONNX metadata and tooling
- Better post-processing and integration examples
- Focus on production-ready inference pipelines

Ease of deployment is considered just as important as raw model performance.

---

## Step 7 – COCO Pretraining (Optional / Long-Term)
COCO pretraining is considered a **potential long-term addition**, with some caveats:

- It is resource-intensive
- It introduces additional complexity
- The Edge series will remain the primary focus

If implemented, COCO-pretrained weights will mainly target **Edge models** and be clearly documented.

---

## Final Notes
This roadmap is intentionally flexible.  
Priorities may shift based on:

- Community feedback
- Available time and hardware
- Practical deployment needs

The project favors **measured progress and technical clarity** over feature churn or hype-driven development.
