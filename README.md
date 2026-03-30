# Object Detection with YOLOv5

**Introduction to Neural Networks**
  
African Institute for Mathematical Sciences (AIMS Rwanda) · March 2026

**Authors:** Olusola Timothy Ogundepo · Lillian Mutesi · Lucas Mirija Razafimanantsoa · Alliance Irigenera

---

## Overview

This project implements a **YOLOv5-style object detector from scratch** in PyTorch and trains it on the Pascal VOC 2012 dataset (20 object classes). The work includes:

- A custom YOLO detector (backbone, detection head, loss function) built entirely in PyTorch — no pre-trained weights used for training.
- A data augmentation study comparing a baseline model against an augmented variant.
- A full written report and Beamer presentation.

---

## Repository Structure

```
.
├── group_3_INN2.ipynb            # Main notebook (all cells executed)
├── object_detection.ipynb        # Supporting experiments
├── group3-copy.ipynb             # Draft copy (experiements)
│
├── group_3_INN2.tex              # LaTeX report source
├── group_3_INN2.pdf              # Compiled report
│
├── Group_3_presentation/
│   ├── group_3_presentation.tex  # Beamer presentation source (submitted)
│   └── group_3_presentation.pdf  # Compiled slides
│
├── group_3_INN2_Presentation.tex # Alternative presentation (AIMS theme)
├── group_3_INN2_Presentation.pdf # Compiled alternative slides
│
├── Images/                       # All project images
│   ├── architecture.png          # Model architecture diagram
│   ├── detection.png             # Sample detection output
│   ├── imbalance.png             # Class imbalance chart
│   ├── loss_compare.png          # Baseline vs augmented loss curves
│   ├── yolo_algo.png             # YOLO algorithm illustration
│   └── test_image.jpg            # Sample test image
│
├── dataset.yaml                  # Dataset config (paths, 20 class names)
├── requirements.txt              # Python dependencies
│
├── Pascal VOC 2012.v1/           # Dataset (train/valid images + labels)
├── yolo_voc.pt                   # Trained baseline model weights
├── yolo_voc_aug.pt               # Trained augmented model weights
│
├── output/                       # Detection output images
└── runs/                         # Training run logs
```

---

## Dataset

**Pascal VOC 2012** — 20 object classes, YOLO-format annotations.

| Split      | Images  |
|------------|---------|
| Train      | 13,690  |
| Validation | 3,422   |

> **Note:** Training used a random **500-image subset** for computational efficiency (CPU, batch size 8).

**Classes:** aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, **person**, pottedplant, sheep, sofa, train, tvmonitor.

Class imbalance is present — the `person` class dominates the training distribution.

---

## Model Architecture

A lightweight YOLOv5-style detector built from scratch:

| Component        | Details |
|------------------|---------|
| Input size       | 128 × 128 px |
| Grid             | S = 8 (8 × 8 = 64 cells) |
| Anchors per cell | B = 3 |
| Classes          | C = 20 |
| Output tensor    | 8 × 8 × 75 (75 = 3 × 25) |
| Backbone         | 9 ConvBlocks + 4 MaxPool layers (~1.47M params) |
| Detection head   | 3 ConvBlocks + final Conv2D (~0.55M params) |
| **Total params** | **~2.02M** |

Each **ConvBlock** = `Conv2D → BatchNorm → LeakyReLU(0.1)`.

---

## Loss Function

The YOLO loss has five components:

$$\mathcal{L} = \lambda_{\text{coord}}(\mathcal{L}_{xy} + \mathcal{L}_{wh}) + \mathcal{L}_{\text{obj}} + \lambda_{\text{noobj}}\mathcal{L}_{\text{noobj}} + \mathcal{L}_{\text{cls}}$$

| Weight | Value | Reason |
|--------|-------|--------|
| λ_coord | 5 | Penalise localisation errors more |
| λ_noobj | 0.05 | Most cells (~97%) are empty |

Square root is applied to width/height to make errors scale-invariant.

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Image size     | 128 × 128 |
| Batch size     | 8 |
| Epochs         | 10 |
| Optimiser      | Adam |
| Learning rate  | 1e-3 |
| Subset size    | 500 images |
| Device         | CPU |
| Random seed    | 42 |

Saved weights: `yolo_voc.pt` (baseline), `yolo_voc_aug.pt` (augmented).

---

## Results

| Set                          | Mean YOLO Loss |
|------------------------------|----------------|
| Training subset (baseline)   | ≈ 12.4128      |
| Validation — full (baseline) | ≈ 12.6849      |
| Training subset (augmented)  | ≈ 11.1727      |
| Validation — full (augmented)| ≈ 13.2454      |

Training loss decreased from **23.34 → 12.38** over 10 epochs (**47% reduction**).

---

## Data Augmentation Study

A second model (`model_aug`) was trained with identical architecture and hyperparameters but with augmentation applied at load time:

- `RandomHorizontalFlip(p=0.5)`
- `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)`
- `RandomRotation(degrees=10)`

**Known limitation:** The horizontal flip was applied to the pixel tensor only; bounding-box coordinates were not mirrored (`c_x → 1 - c_x`). This introduces a label mismatch for ~50% of augmented samples. Colour jitter and rotation still provide useful diversity. A correct fix requires a joint image+box augmentation pipeline.

---

## Installation

```bash
git clone <repo-url>
cd Neural-Network-Group3-Project
pip install -r requirements.txt
```

> Python 3.8+ and PyTorch 1.12+ recommended. No GPU required (CPU training supported).

---

## Usage

Open and run `group_3_INN2.ipynb` in Jupyter. The notebook is fully self-contained and covers:

1. Dataset loading and preprocessing  
2. Model definition (backbone, head, loss)  
3. Training loop (baseline + augmented)  
4. Evaluation and loss curves  
5. Detection visualisation on validation images  

---

## Limitations

1. Small training subset (500 / 13,690 images)
2. Low resolution (128 × 128) — small objects are missed
3. CPU-only training forced architectural simplifications
4. Only 10 epochs — model did not fully converge
5. Horizontal flip label mismatch in augmentation
6. Class imbalance biases detections toward `person`
7. Only YOLO loss reported — no mAP@0.5

---

## Possible Improvements

- Train on all 13,690 images with GPU + 416 × 416 resolution
- Fix horizontal flip with a joint image+box transform
- Anchor clustering (k-means on ground-truth boxes)
- Learning rate scheduler (cosine annealing)
- Report mAP@0.5 for standard benchmark comparison
- Focal loss to address class imbalance
- Transfer learning from a pretrained backbone

---

## Files Compiled from LaTeX

| File | Description |
|------|-------------|
| `group_3_INN2.pdf` | Written report |
| `Group_3_presentation/group_3_presentation.pdf` | Submitted Beamer slides |
| `group_3_INN2_Presentation.pdf` | Alternative AIMS-themed slides |
