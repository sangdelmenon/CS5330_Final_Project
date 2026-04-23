# Presentation Script
## Real-Time Object Recognition and AR Overlay
### CS5330 Final Project, Sangeeth Deleep Menon and Raj Gupta

---

### Slide 1: Title

Hi, we are Sangeeth Deleep Menon and Raj Gupta presenting our CS5330 final project: Real-Time Object Recognition and AR Overlay. The idea is to use deep learning to recognize everyday objects from a webcam and draw augmented reality graphics on top of them, with no printed marker of any kind required.

---

### Slide 2: Motivation

Traditional AR systems need a physical anchor to work. In Project 4 we used a chessboard to estimate camera pose and render 3D graphics. That works well but the target always has to be in the scene.

Project 5 gave us deep learning, CNNs and Vision Transformers trained to classify images.

The question we wanted to answer was: can we combine these two things so that any object the network already knows becomes its own AR anchor? That is what this project does, and the answer is yes.

---

### Slide 3: Data and Pipeline

The dataset has 10 classes and 1419 images total. Five classes, book, cup, keyboard, pen, and phone, were captured with a webcam tool we built. The other five, glasses, headphones, laptop, PS5 controller, and tablet, were downloaded from Bing image search.

The pipeline has four steps. First, a center region of interest is cropped and classified by MobileNetV2. Second, a five-frame majority-vote buffer smooths out flickering, and an entropy check labels uncertain frames as Unknown. Third, a fixed-depth pinhole camera model back-projects the ROI into 3D space at 0.5 meters. Fourth, we use OpenCV's projectPoints to draw a 3D wireframe box and a floating label tag showing the class name and confidence.

The key takeaway from the data is that webcam-captured classes consistently outperformed the web-only ones. Data quality matters more than quantity.

---

### Slide 4: Architectures and Results

We implemented three architectures.

ObjectCNN is a three-layer convolutional network trained from scratch on 64 by 64 images. It hit 79.7% on the 5-class webcam dataset, showing that a simple CNN works well when the data is clean and consistent.

ObjectViT is a minimal Vision Transformer with 8 by 8 patch embeddings and four transformer layers, also trained from scratch. It reached 43.2% on the full 10-class dataset. This is expected behavior. Transformers trained from scratch on small datasets struggle because they lack the pretrained visual features that make them powerful.

MobileNetV2 uses an ImageNet pretrained backbone with the classifier head replaced and fine-tuned. It achieved 88.8% on the full 10-class dataset and was chosen as the deployed model. The live pipeline runs at 20 to 30 frames per second on an Apple Silicon Mac with MPS acceleration.

---

### Slide 5: Where It Struggles

The confusion matrix shows where the model struggles most. Tablet is the weakest class at 70% precision. Flat rectangular devices are hard to separate from phones and laptops at certain angles and zoom levels. Headphones has perfect precision but only 63.6% recall, meaning the model is cautious about predicting it but often misses it.

The main driver of errors is visual ambiguity at class boundaries, not data source alone. PS5 controller finished at 95% on both metrics and glasses recovered to 87.5%, both trained on web data only. So it is not a clean webcam-versus-web story.

---

### Slide 6: Demo Capture

This screenshot was captured during a live demo session. A PS5 DualSense controller is the dominant object in the center region of interest and the model predicts ps5_controller at 85% confidence.

The green 3D wireframe box is cleanly anchored to the object with visible front and back faces connected by pillars. The label tag shows the class name and confidence above the front face. The AR rendering pipeline is robust: the box always appears at the correct position and scale relative to whatever is in the ROI, regardless of whether the classification itself is right.

The session was running at 20 to 30 FPS with a confidence threshold of 0.50.

---

### Slide 7: Summary

To recap the numbers: 10 object classes, 88.8% test accuracy, 20 to 30 FPS live with no printed markers required.

Two limitations remain. The system only looks at the center of the frame so objects off to the side are missed. The fixed 0.5 meter depth assumption means the box scale can be off at very different distances.

Two earlier failure modes have been solved. Prediction flickering is handled by the five-frame majority-vote buffer. The missing background class is handled by entropy-based rejection that shows Unknown when nothing familiar is in view.

Thank you.

---

*Total estimated speaking time: 8 to 10 minutes at a comfortable pace.*
