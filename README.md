# 🧠 Brain Tumor MRI Classification — Deep Learning with Custom Image Processing

A deep learning project for classifying brain tumors from MRI scans into 4 categories using transfer learning and custom-built image processing pipelines. Five CNN architectures are benchmarked against four preprocessing techniques to find the optimal combination for medical image classification.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Tumor Classes](#tumor-classes)
- [Project Structure](#project-structure)
- [Image Processing Pipeline](#image-processing-pipeline)
- [Models](#models)
- [Results](#results)
- [Getting Started](#getting-started)
- [Training Details](#training-details)
- [Dataset](#dataset)

---

## Overview

This project investigates how custom image preprocessing affects the accuracy of deep learning models on brain MRI classification. It implements four handcrafted preprocessing filters from scratch (no `cv2.Sobel()` shortcuts — kernels are applied manually) and benchmarks them across a VGG16 baseline. The best-performing model, a custom CNN trained on LoG+CLAHE preprocessed images, achieves **95.19% accuracy** and a **0.9809 AUC-ROC**.

**Key contributions:**
- Custom `CropBrainContour` class using Gaussian blur → threshold → erode → dilate → BFS connected-component analysis to isolate the brain region
- Four manually implemented preprocessing filters: Sobel edge detection, Laplacian of Gaussian, CLAHE, and Gamma Correction
- Systematic comparison of preprocessing techniques across VGG16
- Full benchmarking of VGG16, DenseNet121, EfficientNetV2-S, ResNet50, and a custom CNN architecture

---

## Tumor Classes

| Class | Description |
|-------|-------------|
| `glioma` | Tumors arising from glial cells, often aggressive |
| `meningioma` | Tumors of the meninges (brain/spine lining), usually benign |
| `notumor` | Healthy brain MRI scans (no tumor present) |
| `pituitary` | Tumors in the pituitary gland |

---

## Project Structure

```
brain-tumor-mri-classifier/
├── Final_Image_Processing_Project.ipynb   # Main notebook (all experiments)
├── custom_cnn.pth                         # Saved Custom CNN weights
├── README.md
└── dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

---

## Image Processing Pipeline

Every image passes through `CropBrainContour` first, which isolates the brain region before any preprocessing filter is applied.

### 1. CropBrainContour (Applied to All Pipelines)

A fully custom contour-detection class that:
1. Converts to grayscale
2. Applies a hand-coded 3×3 Gaussian blur kernel
3. Thresholds at pixel value 15
4. Erodes (1 iteration) and dilates (2 iterations) using 3×3 min/max filters
5. Uses BFS (Breadth-First Search) to find the largest connected white region
6. Crops the image to that bounding box

This removes background noise and annotation text, ensuring the model only sees the anatomical brain structure.

---

### 2. Preprocessing Techniques Compared

#### Sobel Edge Detection
Applies hand-coded horizontal and vertical gradient kernels to extract structural boundaries and edges of potential tumor masses. Output is a 3-channel grayscale-equivalent edge map.

```
Kernel X:            Kernel Y:
[[-1, 0, 1],         [[-1, -2, -1],
 [-2, 0, 2],          [ 0,  0,  0],
 [-1, 0, 1]]          [ 1,  2,  1]]
```

#### Laplacian of Gaussian (LoG)
Blurs with a Gaussian kernel first, then sharpens fine tissue boundaries by subtracting the Laplacian response. This enhances subtle edges without amplifying high-frequency noise.

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
A fully custom tile-based contrast enhancement implementation:
- Divides the image into an 8×8 grid of tiles
- Clips each tile's histogram at `clip_limit × n_pixels / 256` and redistributes the excess
- Computes per-tile CDFs and applies bilinear interpolation between tile lookup tables
- Processes each RGB channel independently

#### LoG + CLAHE (Best Performing)
Chains LoG (for boundary sharpening) followed by CLAHE (for local contrast enhancement). Improves visibility of low-contrast tumour regions without globally amplifying noise.

#### Gamma Correction
Applies `output = 255 × (input/255)^γ` with `γ = 2.2` to standardize brightness and contrast across MRI scans from different machines.

---

## Models

| Model | Architecture | Optimizer | Frozen Layers |
|-------|-------------|-----------|---------------|
| **VGG16** | 13 conv + 3 FC layers | SGD (lr=0.001, momentum=0.9) | All `features` layers |
| **DenseNet121** | Dense blocks with skip connections | Adam (lr=0.001) | All `features` layers |
| **EfficientNetV2-S** | Fused-MBConv blocks | SGD (lr=0.001, momentum=0.9) | All feature layers |
| **ResNet50** | Residual blocks | SGD (lr=0.001, momentum=0.9) | All layers except `fc` |
| **Custom CNN** | 5× Conv-BN-ReLU-Pool + 2 FC layers | Adam (lr=0.001, weight_decay=1e-4) | None (trained from scratch) |

All pretrained models use ImageNet weights with only the final classification head replaced and fine-tuned for 4 classes. The Custom CNN is trained from scratch with a `ReduceLROnPlateau` scheduler.

---

## Results

### Preprocessing Technique Comparison (VGG16 backbone)

| Technique | Accuracy (%) | F1-Score | Precision | Recall | AUC-ROC |
|-----------|-------------|----------|-----------|--------|---------|
| **LoG + CLAHE** | **95.19** | **0.9511** | **0.9534** | **0.9507** | **0.9809** |
| Gamma Correction | 94.06 | 0.9401 | 0.9418 | 0.9395 | 0.9715 |
| LoG | 93.44 | 0.9338 | 0.9352 | 0.9331 | 0.9671 |
| Sobel | 88.31 | 0.8824 | 0.8849 | 0.8818 | 0.9398 |

### Model Architecture Comparison (standard preprocessing)

| Model | Test Accuracy (%) | F1-Score | AUC-ROC |
|-------|------------------|----------|---------|
| **Custom CNN** | **92.88** | **0.9286** | **0.9811** |
| ResNet50 | 83.19 | 0.8317 | 0.9540 |
| VGG16 | — | — | — |
| DenseNet121 | — | — | — |
| EfficientNetV2-S | — | — | — |

> **Key finding:** LoG+CLAHE preprocessing on VGG16 outperforms all other combinations. The Custom CNN (trained from scratch) surprisingly outperforms ResNet50 on this dataset, likely because the lightweight architecture avoids overfitting on the relatively small medical imaging dataset.

### Custom CNN — Per-Class Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| glioma | 0.91 | 0.93 | 0.92 |
| meningioma | 0.92 | 0.91 | 0.91 |
| notumor | 0.96 | 0.97 | 0.96 |
| pituitary | 0.93 | 0.91 | 0.92 |

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended; falls back to CPU automatically)
- Google Colab or local Jupyter environment

### Installation

```bash
pip install torch torchvision
pip install opencv-python-headless imutils
pip install scikit-learn matplotlib pillow
```

### Dataset Setup

The notebook expects the dataset at:
```
/content/drive/MyDrive/Research_Project_2026/dataset/
├── Training/   (glioma/, meningioma/, notumor/, pituitary/)
└── Testing/    (glioma/, meningioma/, notumor/, pituitary/)
```

The original dataset is the [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Place your `kaggle.json` API key in the `SHARED_PATH` directory before running.

### Run the Notebook

```bash
# On Google Colab (recommended for GPU access)
# 1. Mount Google Drive
# 2. Upload the notebook to Colab
# 3. Run all cells in order

# Or locally:
jupyter notebook Final_Image_Processing_Project.ipynb
```

### Load the Saved Model

```python
import torch
from torchvision import transforms

model = CustomCNN(num_classes=4)
model.load_state_dict(torch.load('custom_cnn.pth'))
model.eval()
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Input size | 224 × 224 |
| Batch size | 32 |
| Train/Val split | 80% / 20% |
| Max epochs | 30–35 |
| Early stopping patience | 5 epochs |
| Min delta | 0.001 |
| Random seed | 42 |
| Normalization | ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |

**Data augmentation (training only):**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.1, contrast=0.1)

**Custom CNN scheduler:** `ReduceLROnPlateau` — halves learning rate after 2 epochs without validation loss improvement.

---

## Dataset

- **Source:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total test images:** 1,600 (400 per class, balanced)
- **Image type:** T1-weighted contrast-enhanced MRI scans
- **Format:** JPEG, variable resolution → resized to 224×224

---

## License

This project is for academic and educational purposes. MRI images are sourced from a publicly available Kaggle dataset. No patient-identifiable information is present in the dataset.
