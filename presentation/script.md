# Presentation Script
## Real-Time Object Recognition and AR Overlay
### CS5330 Final Project — Sangeeth Deleep Menon

---

### Slide 1 — Title

Hi, my name is Sangeeth Deleep Menon and this is my CS5330 final project.
The project is called Real-Time Object Recognition and AR Overlay.
The core idea is to use deep learning to recognize everyday objects from a webcam feed, and then draw augmented reality graphics directly on top of the detected object — with no printed marker required.

---

### Slide 2 — The Problem

Traditional augmented reality systems need a physical anchor to work. In Project 4 of this course we used a chessboard calibration target to estimate camera pose and render 3D graphics. That works well, but it means you always need the target in the scene.

Project 5 gave us deep learning — CNNs and Vision Transformers trained to classify images.

The question I wanted to answer was: can we combine these two things so that any object the network already knows becomes its own AR anchor? That is exactly what this project does.

---

### Slide 3 — System Overview

The pipeline has three steps.

First, we crop a square region from the center of the webcam frame and run it through MobileNetV2 to get a predicted class and a confidence score.

Second, if the confidence is above the threshold, we estimate a rough object pose. We use a fixed-depth pinhole camera model — the object is assumed to be at about half a meter from the camera — and we back-project the bounding box into 3D space.

Third, we use OpenCV's projectPoints function to map the eight corners of a 3D box back into pixel coordinates, and we draw the wireframe box along with a floating label tag showing the class name and confidence.

---

### Slide 4 — Dataset

The dataset has 10 classes and 1,419 images total.

Five classes — book, cup, keyboard, pen, and phone — were captured directly with a webcam tool I built, collecting images from different angles, distances, and lighting conditions.

The other five classes — glasses, headphones, laptop, PS5 controller, and tablet — were downloaded from Bing image search using the icrawler library.

The split is 70% training, 15% validation, and 15% test, with a fixed random seed so the results are fully reproducible. Training images were augmented with horizontal flips, color jitter, and random rotations.

The key observation from the results is that webcam-captured classes consistently outperformed web-only classes. Data quality matters more than quantity.

---

### Slide 5 — Model Architectures

I implemented and compared three architectures.

The first is ObjectCNN — a simple three-layer convolutional network trained from scratch on 64-by-64 RGB images. It achieved near-perfect accuracy on the original five clean webcam classes, which shows that a simple CNN is more than capable when the data is consistent.

The second is ObjectViT — a minimal Vision Transformer with 8-by-8 patch embeddings and four transformer encoder layers, also trained from scratch. It converges more slowly and needs more data to generalize well.

The third — and the one deployed in the live system — is MobileNetV2 with transfer learning. The ImageNet-pretrained backbone provides strong visual features that generalize to new classes even with only 100 to 170 training images per class. This one achieved 84.1% accuracy on the full 10-class dataset.

---

### Slide 6 — Results

Three numbers tell the story.

84.1% test accuracy overall on the 10-class held-out set. The system runs at 30 frames per second on an Apple Silicon Mac using MPS acceleration, well above the 15 FPS minimum for a smooth AR experience. And the strongest individual class was phone, where 29 out of 30 test images were classified correctly.

The training curves show the model converging steadily through the first 8 epochs and then leveling off, with mild overfitting visible in the later epochs.

---

### Slide 7 — Where It Struggles

The confusion matrix shows the full picture.

The weakest class is glasses. The web crawl returned a mix of eyeglasses and drinking glasses — both under the same label — which introduced noise the model could not overcome.

Tablet and phone are frequently confused because they are both flat rectangles and look nearly identical at certain angles and zoom levels. The AR screenshot I will show next is actually a real example of this: a tablet being classified as a phone.

The root cause for all five weak classes is the same — they were trained only on web images. The five webcam-captured classes all had near-perfect precision and recall.

---

### Slide 8 — Live Demo

This screenshot was captured during the live demo. A tablet is standing upright on a desk, and the model predicts phone at 84% confidence.

Even though the classification is wrong, the 3D wireframe box and label are drawn correctly around the detected region. This is an important point: the AR rendering pipeline is robust regardless of whether the classification itself is correct. The box always appears properly positioned and scaled relative to the object in the frame.

The system was running at about 29.8 FPS during this session with a confidence threshold of 0.60.

---

### Slide 9 — Limitations and Next Steps

There are four current limitations worth mentioning.

The first is the fixed center ROI — the system only looks at the center of the frame, so objects off to the side are ignored. The second is fixed depth — the 0.5-meter assumption means the 3D box will appear incorrectly scaled if the object is much closer or further away. Third, there is no background class, so the model always outputs its best guess even when nothing recognizable is in frame. Finally, predictions can flicker between frames when confidence is near the threshold.

The easiest fixes are: collect webcam images for the five web-only classes, which would likely push accuracy above 90%; add a short history buffer of five frames to smooth out flickering; add a background class; and replace the fixed ROI with a sliding-window or YOLO-based detector.

---

### Slide 10 — Summary

To summarize: this project built a working real-time markerless AR system that combines the deep learning work from Project 5 with the AR rendering pipeline from Project 4.

The system recognizes 10 everyday object classes at 30 frames per second with 84.1% accuracy, and it requires zero printed markers. Any object the network knows becomes its own AR anchor.

Thank you.

---

*Total estimated speaking time: ~8–10 minutes at a comfortable pace.*
