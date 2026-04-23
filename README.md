# CS5330 Final Project: Real-Time Object Recognition and Augmented Reality Fusion

**Students:** Sangeeth Deleep Menon (NUID: 002524579) and Raj Gupta (NUID: 002068701)
**Program:** MSCS Boston, Section 03 (CRN 40669, Online)
**Course:** CS5330 Pattern Recognition and Computer Vision, Spring 2026

**Demo video:** *(add YouTube/Google Drive URL here after recording)*
**Dataset:** *(add Google Drive URL here if uploading)*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Setup and Dependencies](#setup-and-dependencies)
5. [Step-by-Step Usage](#step-by-step-usage)
   - [Step 1: Collect Training Data](#step-1-collect-training-data)
   - [Step 2: Train the Model](#step-2-train-the-model)
   - [Step 3: Run the Live AR System](#step-3-run-the-live-ar-system)
6. [What Has Been Done](#what-has-been-done)
7. [What Is Expected (Deliverables)](#what-is-expected-deliverables)
8. [Architecture Details](#architecture-details)
   - [ObjectCNN](#objectcnn)
   - [ObjectViT](#objectvit)
   - [AR Rendering Pipeline](#ar-rendering-pipeline)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Known Limitations and Future Work](#known-limitations-and-future-work)
11. [Connections to Prior Projects](#connections-to-prior-projects)

---

## Project Overview

This project builds a **markerless real-time object recognition and augmented reality system** by combining two major themes from earlier in the course:

- **Project 4:** AR rendering pipeline using a calibration target (chessboard) to estimate camera pose and overlay 3D graphics
- **Project 5:** Deep learning for image recognition using CNNs and Vision Transformers trained on MNIST and Greek letters

The key novelty is that the system does not require any printed marker or calibration target. Any everyday object that the network has learned to recognise becomes its own AR anchor. When the classifier detects a known object with sufficient confidence, the system:

1. Estimates a rough object pose using the detected bounding box and a fixed-depth pinhole camera model
2. Projects a 3D wireframe box onto the object using `cv2.projectPoints`
3. Renders a floating semi-transparent label tag with the class name and confidence score above the object, all in real time at interactive frame rates

---

## How It Works

```
Webcam frame
     │
     ▼
 Centre ROI crop (configurable size)
     │
     ▼
 Preprocess: resize to 64×64 RGB → normalise
     │
     ▼
 ObjectCNN (or ObjectViT) → class index + confidence
     │
     ├─── confidence < threshold ──► draw greyed-out "?" label
     │
     └─── confidence ≥ threshold ──► 3D AR overlay
                                          │
                                          ├─ 2D bounding box
                                          ├─ 3D wireframe box (cv2.projectPoints)
                                          └─ floating label tag (class + %)
```

The camera matrix is estimated from the image dimensions (focal length ≈ 0.85 × max(width, height)), and the object is assumed to be at a fixed depth of 0.5 m. The 3D box dimensions are back-projected from pixel size at that depth, giving a consistent AR overlay without any physical calibration.

---

## Project Structure

```
Final_Project/
│
├── model.py            # CNN and ViT architecture definitions
├── collect_data.py     # Interactive webcam data-collection tool
├── train.py            # Training, validation, and test evaluation script
├── recognize_ar.py     # Main live AR recognition pipeline
│
├── data/               # Created by collect_data.py — one sub-folder per class
│   ├── cup/
│   ├── phone/
│   ├── book/
│   ├── keyboard/
│   └── pen/
│
├── object_model.pth    # Saved best model checkpoint (created by train.py)
│
├── training_curves.png     # Loss and accuracy plots (created by train.py)
├── confusion_matrix.png    # Per-class confusion heatmap (created by train.py)
├── ar_screenshot.png       # Saved with S key during live demo
│
├── final_project_proposal.md
├── final_project_proposal.pdf
└── README.md
```

---

## Setup and Dependencies

The project requires Python 3.10+ and the following packages:

```
torch >= 2.0
torchvision >= 0.15
opencv-python >= 4.5
numpy
matplotlib
Pillow
```

Install with pip:

```bash
pip install torch torchvision opencv-python numpy matplotlib Pillow
```

If you are on an Apple Silicon Mac (MPS acceleration):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

PyTorch will automatically use MPS, CUDA, or CPU in that priority order. All scripts print the active device at startup.

---

## Step-by-Step Usage

### Step 1: Collect Training Data

```bash
python collect_data.py
```

Or specify your own classes:

```bash
python collect_data.py --classes cup phone book keyboard pen
python collect_data.py --classes mug laptop pen notebook stapler --camera 1
```

**Controls inside the collection window:**

| Key | Action |
|-----|--------|
| `SPACE` | Capture one image for the active class |
| `A` | Toggle auto-capture (grabs one frame every 12 frames automatically) |
| `0`–`9` | Switch the active class by index |
| `Q` | Quit and print a summary of images collected |

**Tips for good data:**
- Collect **at least 60–80 images per class** for acceptable accuracy
- Vary **distance, angle, rotation, and lighting** while capturing
- Auto-capture mode (`A`) while slowly rotating an object works well
- Make sure each object fills roughly the same fraction of the green ROI box as it will in the live demo

Images are saved to `data/<class_name>/0001.jpg`, `0002.jpg`, etc.

---

### Step 2: Train the Model

```bash
python train.py                              # CNN, 20 epochs, lr=0.001
python train.py --model vit --epochs 30      # Vision Transformer instead
python train.py --epochs 30 --batch 64 --lr 5e-4
```

**What train.py does:**

1. Loads all images from `data/` using `torchvision.datasets.ImageFolder`
2. Splits into **70% train / 15% val / 15% test** (reproducible with seed 42)
3. Applies augmentation to training images only (horizontal flip, colour jitter, random rotation ±15°)
4. Trains with Adam + cosine annealing LR schedule
5. Saves the epoch with the **best validation accuracy** to `object_model.pth`
6. After training, reloads the best checkpoint and reports:
   - Overall test accuracy
   - Per-class **precision and recall**
7. Saves `training_curves.png` and `confusion_matrix.png`

**Output example:**

```
Epoch   tr_loss    tr_acc%    vl_loss    vl_acc%
----------------------------------------------------
  1      0.8312      72.4%     0.7104      76.2%
  5      0.3201      88.9%     0.3450      87.5%  <- best
 ...
Best validation accuracy: 91.3%

Per-class results:
Class            Precision      Recall
----------------------------------------
cup                  94.1%       92.0%
phone                89.5%       91.3%
book                 93.0%       88.7%
keyboard             90.2%       93.5%
pen                  88.8%       90.1%
----------------------------------------
Mean                 91.1%       91.1%
```

The saved checkpoint (`object_model.pth`) stores the model weights, the class name list, the architecture type, and the image size so that `recognize_ar.py` can reconstruct the model exactly without any additional configuration.

---

### Step 3: Run the Live AR System

```bash
python recognize_ar.py
python recognize_ar.py --conf 0.65          # raise confidence threshold
python recognize_ar.py --roi 0.45           # smaller ROI box
python recognize_ar.py --camera 1           # use a different camera
```

**Controls during live demo:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot to `ar_screenshot.png` |

**What you will see:**

- A live webcam feed with a centred detection ROI
- When a known object is held in the ROI with confidence ≥ threshold:
  - A **coloured 2D border** around the ROI
  - A **3D wireframe box** projected onto the object (front and back faces + connecting edges)
  - A **floating label tag** above the box showing the class name and confidence percentage
- When confidence is below the threshold: a grey `?` label with the raw score
- **FPS counter** in the top-left corner

---

## What Has Been Done

| Component | Status | Notes |
|-----------|--------|-------|
| `model.py` — ObjectCNN | Done | 3-conv BN-ReLU CNN, 3×64×64 input, ~1.2M params |
| `model.py` — ObjectViT | Done | Patch-8 ViT, 4 transformer layers, same I/O |
| `collect_data.py` | Done | Live ROI capture with auto-capture and class switching |
| `train.py` | Done | 70/15/15 split, augmentation, cosine LR, precision/recall, confusion matrix |
| `recognize_ar.py` | Done | Real-time classify + 3D wireframe box + floating label tag |
| Camera model (AR) | Done | Estimated pinhole matrix, fixed Z=0.5m, `cv2.projectPoints` |
| FPS display | Done | Exponential moving average shown live |
| Screenshot save | Done | Press S in live window |

---

## What Is Expected (Deliverables)

The following items are expected for the final submission:

### Code and Demo
- [ ] **Working live demo** — `recognize_ar.py` runs at ≥ 15 FPS on a laptop (MPS/CUDA preferred)
- [ ] **Trained model checkpoint** — `object_model.pth` included in submission
- [ ] All four scripts (`model.py`, `collect_data.py`, `train.py`, `recognize_ar.py`) present and runnable

### Dataset
- [ ] At least **5 object classes** collected
- [ ] At least **60 images per class** (300+ total)
- [ ] Dataset directory (`data/`) included or described

### Quantitative Evaluation
- [ ] **Overall test accuracy** reported
- [ ] **Per-class precision and recall** table produced by `train.py`
- [ ] `confusion_matrix.png` saved and included in report
- [ ] **FPS measurement** noted (timing benchmark confirming real-time performance)

### Report (written separately)
- [ ] Description of the CNN and ViT architectures with comparison
- [ ] Training curves (`training_curves.png`) included and discussed
- [ ] Confusion matrix discussed — which classes are confused and why
- [ ] AR rendering approach explained (camera model, fixed-depth projection)
- [ ] Discussion of limitations (fixed ROI, no true depth, single-object detection)
- [ ] At least one AR screenshot (`ar_screenshot.png`) included

### Optional Extensions (for bonus credit / stronger submission)
- [ ] Train both CNN and ViT and compare accuracy vs. inference speed
- [ ] Add a second detection region or support multiple simultaneous objects
- [ ] Replace the fixed ROI with a sliding-window or saliency-based detector
- [ ] Collect a larger dataset (150+ images/class) for higher accuracy

---

## Architecture Details

### ObjectCNN

```
Input: 3 × 64 × 64 (RGB, normalised with ImageNet mean/std)

Conv2d(3→32, 3×3, pad=1) → BatchNorm → ReLU → MaxPool(2)   → 32 × 32 × 32
Conv2d(32→64, 3×3, pad=1) → BatchNorm → ReLU → MaxPool(2)  → 64 × 16 × 16
Conv2d(64→128, 3×3, pad=1) → BatchNorm → ReLU → MaxPool(2) → 128 × 8 × 8

Flatten → 8192
Linear(8192 → 256) → ReLU → Dropout(0.4)
Linear(256 → num_classes) → log_softmax
```

BatchNorm is used after each convolution (unlike the MNIST CNN from Project 5) because training on a small custom dataset benefits from normalisation within the network. Dropout(0.4) on the fully-connected layer provides regularisation.

### ObjectViT

```
Input: 3 × 64 × 64

PatchEmbed: Conv2d(3→128, 8×8, stride=8) → 64 patches of dim 128
Prepend CLS token → sequence length 65
Add learned positional embeddings
4 × TransformerEncoderLayer(d=128, heads=4, ff=512, dropout=0.1)
LayerNorm
Extract CLS token → Linear(128 → num_classes) → log_softmax
```

The ViT is trained from scratch (no ImageNet pre-training), so it typically requires more data and more epochs than the CNN to converge. On small datasets (< 500 images total) the CNN usually outperforms the ViT.

### AR Rendering Pipeline

The AR overlay uses a **fixed-depth pinhole camera model**:

1. **Camera matrix estimation:**
   ```
   f  = max(frame_width, frame_height) × 0.85
   K  = [[f, 0, cx],
         [0, f, cy],
         [0, 0,  1]]
   ```

2. **3D box construction:** The detected ROI (x1, y1, x2, y2) is back-projected to 3D assuming Z = 0.5 m. The box width W and height H follow from the pinhole equation (`W = pixel_width × Z / f`). Box depth D = 0.45 × min(W, H).

3. **Projection:** `cv2.projectPoints` maps all 8 box corners to pixel coordinates using the identity rotation/translation (object coordinate frame aligned with camera).

4. **Drawing:** 12 edges (front face, back face, four pillars) are drawn as coloured lines. A unique colour is assigned per class from a fixed palette.

5. **Label tag:** A semi-transparent filled rectangle (alpha-blended) is placed above the top edge of the projected front face, with the class name and confidence overlaid in dark text.

---

## Evaluation Metrics

**Classification:**
- **Accuracy** — fraction of test images classified correctly
- **Precision (per class)** — TP / (TP + FP): of all frames predicted as class C, how many actually were C
- **Recall (per class)** — TP / (TP + FN): of all frames that were class C, how many were detected

**Performance:**
- **FPS** — measured as exponential moving average during the live loop; target ≥ 15 FPS on a standard laptop CPU, ≥ 30 FPS with MPS/CUDA

---

## Known Limitations and Future Work

| Limitation | Description |
|------------|-------------|
| Fixed-centre ROI | Only classifies one region per frame; objects off-centre are not detected |
| Fixed depth | Z = 0.5 m is a rough assumption; the 3D box will appear incorrectly scaled at very different distances |
| Small training set | Accuracy depends heavily on dataset quality; the five web-only classes would benefit from additional webcam images |
| Single object | Multiple simultaneous objects are not handled; a detector-based approach such as YOLO would be needed |

**Features implemented to address common failure modes:**

| Feature | Description |
|---------|-------------|
| Temporal smoothing | A 5-frame majority-vote history buffer prevents label flickering at decision boundaries |
| Entropy-based rejection | Frames where the probability distribution is too flat are labelled "Unknown", removing the need for a dedicated background class |

---

## Connections to Prior Projects

| Feature | Origin | Extension in Final Project |
|---------|--------|-----------------------------|
| CNN architecture | Project 5 Task 1 | Adapted from MNIST (1ch, 28×28) → RGB (3ch, 64×64), added BatchNorm |
| Architecture experiment | Project 5 Task 5 | Best-performing config informed choice of filter counts and FC size |
| Vision Transformer | Project 5 Task 4 | Rebuilt as minimal patch-8 ViT; same training loop as P5 transformer |
| Live webcam loop | Project 5 Extension | Generalised from single-digit ROI to multi-class object ROI |
| AR overlay + camera model | Project 4 | Replaces chessboard pose estimation with fixed-depth back-projection; uses same `cv2.projectPoints` API |
