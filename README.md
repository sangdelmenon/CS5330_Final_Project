# CS5330 Final Project: Real-Time Object Recognition and Augmented Reality Fusion

**Students:** Sangeeth Deleep Menon (NUID: 002524579) and Raj Gupta (NUID: 002068701)\
**Program:** MSCS Boston, Section 03 (CRN 40669, Online)\
**Course:** CS5330 Pattern Recognition and Computer Vision, Spring 2026

**Presentation video:** https://drive.google.com/file/d/1l1NJLrnBtZBt5Nko6NsMIqZLKrwrT2dz/view?usp=sharing
**Dataset:** https://drive.google.com/drive/folders/1r2CQKL2AjviCOPXwH1MrJZTH9IFINoHA?usp=drive_link

---

## Project Description

This project builds a markerless real-time object recognition and augmented reality system by combining two major themes from earlier in the course. Project 4 covered how to render 3D graphics onto a physical scene using a chessboard calibration target to estimate camera pose. Project 5 covered deep learning for image recognition using CNNs and Vision Transformers. The key novelty here is that no printed marker or calibration target is needed. Any everyday object that the network has learned to recognize becomes its own AR anchor.

When the classifier detects a known object with sufficient confidence, the system estimates a rough object pose using the detected bounding box and a fixed-depth pinhole camera model, projects a 3D wireframe box onto the object using cv2.projectPoints, and renders a floating semi-transparent label tag with the class name and confidence score above the object, all in real time.

---

## Project Structure

```
Final_Project/
├── model.py            # CNN, ViT, and MobileNetV2 architecture definitions
├── collect_data.py     # Interactive webcam data-collection tool
├── train.py            # Training, validation, and test evaluation script
├── recognize_ar.py     # Main live AR recognition pipeline
├── download_images.py  # Bing image downloader for web-only classes
├── data/               # One sub-folder per class (created by collect_data.py)
├── Figures/            # Training curves and confusion matrices
└── object_model.pth    # Saved best model checkpoint (created by train.py)
```

---

## Setup

The project requires Python 3.10 or later with the following packages:

```
torch >= 2.0
torchvision >= 0.15
opencv-python >= 4.5
numpy
matplotlib
Pillow
icrawler
```

Install with pip:

```bash
pip install torch torchvision opencv-python numpy matplotlib Pillow icrawler
```

PyTorch will automatically use MPS (Apple Silicon), CUDA, or CPU in that priority order. All scripts print the active device at startup.

---

## Usage

### Step 1: Collect Training Data

```bash
python collect_data.py
python collect_data.py --classes cup phone book keyboard pen
```

Press SPACE to capture one image, A to toggle auto-capture, 0 through 9 to switch between classes, and Q to quit. Aim for at least 80 images per class, varying the distance, angle, and lighting as you go.

### Step 2: Train the Model

```bash
python train.py                         # CNN, 20 epochs
python train.py --model mobilenet       # MobileNetV2 (best accuracy)
python train.py --model vit --epochs 20 # Vision Transformer
```

The script splits the dataset 70/15/15 (train/val/test) with a fixed seed of 42, applies augmentation to training images only, trains with Adam and cosine annealing, saves the best checkpoint by validation accuracy, and reports overall accuracy and per-class precision and recall on the held-out test set. Training curves and a confusion matrix are saved to the Figures/ directory. Note that all three architectures save to the same object_model.pth filename by default, so running multiple architectures in sequence will overwrite the previous checkpoint. Rename the file between runs if you want to keep all three.

### Step 3: Run the Live AR System

```bash
python recognize_ar.py
python recognize_ar.py --conf 0.65          # raise confidence threshold
python recognize_ar.py --sliding            # sliding-window detection
python recognize_ar.py --roi 0.45 --camera 1
```

Press Q to quit and S to save a screenshot. The live window shows a detection ROI, and when a known object is held in view with enough confidence, a coloured 3D wireframe box and floating label tag are drawn over it. The FPS counter appears in the top-left corner.

---

## Architecture Summary

Three architectures were implemented and compared.

**ObjectCNN** is a three-layer convolutional network trained from scratch. It accepts 3x64x64 RGB images. Each convolutional layer is followed by batch normalization, ReLU, and max pooling. A fully connected layer with 256 neurons and 40% dropout connects to the output.

**ObjectViT** is a minimal Vision Transformer also trained from scratch. The image is split into 64 non-overlapping 8x8 patches, each projected into a 128-dimensional embedding. Four transformer encoder layers with 4 attention heads and a 512-dimensional feed-forward dimension process the sequence. The class token output goes to the final linear head.

**MobileNetV2** uses an ImageNet pretrained backbone with the final classifier replaced by a 30% dropout layer and a new linear output sized for 10 classes. The backbone is fine-tuned during training with a lower learning rate than the head.

---

## Results

| Model | Classes | Test Accuracy | Input Size |
|---|---|---|---|
| ObjectCNN | 5 (webcam) | 79.7% | 64x64 |
| MobileNetV2 | 10 | 88.8% | 224x224 |

The live pipeline runs at 20 to 30 FPS on an Apple Silicon Mac with MPS acceleration. Cup and keyboard both reached 100% recall. Headphones reached 100% precision. Tablet was the weakest class at 70% precision due to visual similarity with phones and laptops.

---

## Known Limitations

The system classifies only a fixed center region of the frame, so objects off to the side are not detected. The 3D box is back-projected assuming the object is always 0.5 meters from the camera, so it will appear incorrectly scaled at very different distances. A sliding-window mode is available via `--sliding` as an alternative to the center ROI.

Two earlier failure modes have been addressed: prediction flickering is handled by a five-frame majority-vote buffer, and the lack of a background class is handled by entropy-based rejection that displays Unknown when the model sees nothing familiar.

---

## Connections to Prior Projects

The AR rendering pipeline comes from Project 4, adapted to work without any physical target by using the detected bounding box as the pose reference and a fixed-depth assumption in place of a solved pose. The deep learning components come from Project 5, which trained CNNs and Vision Transformers on MNIST and Greek letter datasets. MobileNetV2 with transfer learning goes beyond what was covered in the course and was necessary to achieve strong accuracy on the 10-class dataset with limited training data.
