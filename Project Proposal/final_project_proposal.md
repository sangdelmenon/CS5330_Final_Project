# CS5330 Final Project Proposal
## Real-Time Object Recognition and Augmented Reality Fusion

**Student:** Sangeeth Deleep Menon
**NUID:** 002524579

*Note: Raj Gupta (NUID: 002068701) joined the project after this proposal was submitted. The final report reflects contributions from both team members.*
**Program:** MSCS Boston, Section 03 (CRN 40669, Online)

---

### Project Overview

I want to build a real-time object recognition and augmented reality system that combines the deep learning work from Project 5 with the AR rendering pipeline from Project 4. The system will use a trained convolutional neural network to detect and classify objects from a live webcam feed, then draw AR annotations such as labels, bounding boxes, and simple 3D graphics directly on top of each recognized object. Unlike the AR work in Project 4, this system will not require a printed chessboard or ArUco marker. Instead, any object the network can recognize becomes its own AR anchor.

This is a meaningful step forward from both prior projects. Project 4 needed a physical calibration target to estimate pose, and Project 5 ran recognition offline on still images. Putting them together creates something that feels more like a real application.

---

### Planned Work

**1. Multi-class object recognizer**
I will collect or curate a small dataset of 5 to 10 everyday object classes such as a cup, phone, book, keyboard, and pen. I will train both a CNN and a Vision Transformer on this data and pick whichever performs better as the backbone. This builds directly on the Task 4 and Task 5 experiments from Project 5.

**2. Markerless AR overlay**
Instead of tracking a chessboard, I will use the bounding box from the classifier along with a depth estimate from monocular cues or a fixed camera model to compute a rough object pose. From there I can render a 3D label tag or a simple 3D shape anchored to the recognized object.

**3. Real-time pipeline with a GUI**
The full system will run at interactive frame rates on a standard laptop using PyTorch with MPS or CUDA acceleration. A tkinter or OpenCV window will show the live camera feed along with the detected class name, confidence score, and AR overlays all at once.

**4. Quantitative evaluation**
I will measure per-class precision and recall on a held-out test set. I will also time the full inference and rendering pipeline to confirm the system runs in real time.

---

### Why This Project

This project ties together the two most substantial assignments from the course and extends both in a concrete way. It involves real engineering tradeoffs around model selection, real-time performance, and AR rendering. The end result is a live demo that is easy to show and evaluate, and the scope feels right for a solo project with a clear path from start to finish.
