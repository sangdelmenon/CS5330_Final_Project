# Sangeeth Deleep Menon
# NUID: 002524579
# Raj Gupta (NUID: 002068701)
# CS5330 Final Project - Real-Time AR Object Recognition
# Spring 2026
#
# Loads the trained classifier and runs a live webcam loop.
# For each frame the centre ROI is classified, and if confidence is
# above the threshold the following AR overlays are drawn:
#
#   1. 2D bounding box around the detection ROI
#   2. 3D wireframe box projected onto the object using a fixed-depth
#      pinhole camera model (cv2.projectPoints) – no printed marker needed
#   3. Floating semi-transparent label tag with class name and confidence
#
# Temporal smoothing: predictions are averaged over a short history buffer
# to reduce flicker. Entropy-based rejection flags frames where the model
# is genuinely uncertain as "Unknown" even if one class has the highest score.
#
# Usage:
#   python recognize_ar.py                          # uses object_model.pth
#   python recognize_ar.py --conf 0.6 --roi 0.5
#   python recognize_ar.py --history 7 --entropy-thresh 0.88
#
# Controls:
#   Q - quit
#   S - save screenshot to ar_screenshot.png

import sys
import argparse
import time
from collections import deque, Counter

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import ObjectCNN, ObjectViT, MobileNetV2, IMG_SIZE


_CLASS_COLORS = [
    (  0, 220,   0),
    (  0, 140, 255),
    (200,   0, 200),
    (255, 200,   0),
    (128,   0, 255),
    (  0, 200, 200),
    (255,  80,  80),
    (  0, 200, 255),
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Real-time AR object recognition')
    p.add_argument('--model',  default='object_model.pth')
    p.add_argument('--camera', type=int,   default=0)
    p.add_argument('--conf',   type=float, default=0.50,
                   help='Minimum confidence to show AR overlay (0-1)')
    p.add_argument('--roi',    type=float, default=0.55,
                   help='ROI size as fraction of the shorter image dimension')
    p.add_argument('--history', type=int,  default=5,
                   help='Number of frames to smooth predictions over')
    p.add_argument('--entropy-thresh', type=float, default=0.85,
                   help='Normalised entropy above which the frame is shown as Unknown (0-1)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes    = checkpoint['classes']
    arch       = checkpoint.get('arch', 'cnn')
    img_size   = checkpoint.get('img_size', IMG_SIZE)

    if arch == 'vit':
        model = ObjectViT(len(classes))
    elif arch == 'mobilenet':
        model = MobileNetV2(len(classes))
    else:
        model = ObjectCNN(len(classes))
    model.load_state_dict(checkpoint['model_state'])
    model.eval().to(device)
    return model, classes, img_size


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_roi_box(h, w, fraction):
    """Returns (x1, y1, x2, y2) for a centred square ROI."""
    size = int(min(h, w) * fraction)
    cx, cy = w // 2, h // 2
    return cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2


def build_camera_matrix(w, h):
    """Estimates a pinhole intrinsic matrix. Focal length ≈ 85% of the longer dimension."""
    f = max(w, h) * 0.85
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0   ]], dtype=np.float64)


# ---------------------------------------------------------------------------
# AR rendering
# ---------------------------------------------------------------------------

def project_3d_box(frame, roi_box, camera_matrix, color, thickness=2):
    """
    Draws a 3D wireframe box anchored to the detected ROI at a fixed depth of 0.5 m.
    Returns the projected 2D corner array (shape 8x2).
    """
    x1, y1, x2, y2 = roi_box
    cx_roi = (x1 + x2) / 2.0
    cy_roi = (y1 + y2) / 2.0

    f    = camera_matrix[0, 0]
    cx_c = camera_matrix[0, 2]
    cy_c = camera_matrix[1, 2]
    Z    = 0.5

    W = (x2 - x1) * Z / f
    H = (y2 - y1) * Z / f
    D = min(W, H) * 0.45

    X = (cx_roi - cx_c) * Z / f
    Y = (cy_roi - cy_c) * Z / f

    corners = np.array([
        [X - W/2, Y - H/2, Z - D/2],
        [X + W/2, Y - H/2, Z - D/2],
        [X + W/2, Y + H/2, Z - D/2],
        [X - W/2, Y + H/2, Z - D/2],
        [X - W/2, Y - H/2, Z + D/2],
        [X + W/2, Y - H/2, Z + D/2],
        [X + W/2, Y + H/2, Z + D/2],
        [X - W/2, Y + H/2, Z + D/2],
    ], dtype=np.float32)

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)

    pts, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, dist)
    pts    = pts.reshape(-1, 2).astype(int)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)

    return pts


def draw_label_tag(frame, text, anchor_px, color, bg_alpha=0.72):
    """
    Draws a semi-transparent filled rectangle with dark text.
    anchor_px is the bottom-centre pixel; the tag floats upward from there.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.80
    thickness  = 2
    pad        = 9

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    ax, ay = anchor_px

    x0 = ax - tw // 2 - pad
    y0 = ay - th - baseline - pad * 2
    x1 = ax + tw // 2 + pad
    y1 = ay

    fh, fw = frame.shape[:2]
    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(fw, x1); y1 = min(fh, y1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, cv2.FILLED)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)

    tx = ax - tw // 2
    ty = ay - baseline - pad
    cv2.putText(frame, text, (tx, ty), font, font_scale, (10, 10, 10), thickness)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_roi(model, transform, roi_bgr, device):
    """
    Returns (class_index, confidence, normalised_entropy) for the ROI.

    Normalised entropy is entropy / log(num_classes), so it ranges from 0
    (perfectly certain) to 1 (completely uniform — unknown object).
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(Image.fromarray(roi_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.exp(model(tensor))

    p    = probs[0].cpu().numpy()
    idx  = int(p.argmax())
    conf = float(p[idx])

    entropy     = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(p))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    return idx, conf, norm_entropy


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

def run_live(args):
    device = torch.device('mps'  if torch.backends.mps.is_available()  else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model, classes, img_size = load_model(args.model, device)
    transform = build_transform(img_size)
    print('Classes:', classes)
    print('Confidence threshold: {:.0f}%  |  History: {} frames  |  Entropy threshold: {:.2f}'.format(
        args.conf * 100, args.history, args.entropy_thresh))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Error: cannot open camera', args.camera)
        return

    history   = deque(maxlen=args.history)
    prev_time = time.time()
    fps       = 0.0

    print('\nAR recognition running.  Q = quit   S = screenshot\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        roi_box       = get_roi_box(h, w, args.roi)
        camera_matrix = build_camera_matrix(w, h)
        x1, y1, x2, y2 = roi_box

        roi                    = frame[y1:y2, x1:x2]
        raw_idx, conf, entropy = classify_roi(model, transform, roi, device)

        # Temporal smoothing: pick the most common prediction over recent frames
        history.append(raw_idx)
        cls_idx    = Counter(history).most_common(1)[0][0]
        class_name = classes[cls_idx]
        color      = _CLASS_COLORS[cls_idx % len(_CLASS_COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Entropy-based unknown rejection: high entropy means the model sees
        # nothing it recognises, regardless of the top confidence score
        unknown = entropy > args.entropy_thresh

        if unknown:
            cv2.putText(frame, 'Unknown',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 110), 2)
        elif conf >= args.conf:
            corner_pts = project_3d_box(frame, roi_box, camera_matrix, color)

            top_pts    = corner_pts[:4]
            tag_x      = int(top_pts[:, 0].mean())
            tag_y      = int(top_pts[:, 1].min()) - 8
            tag_y      = max(30, tag_y)
            draw_label_tag(frame,
                           '{} {:.0f}%'.format(class_name, conf * 100),
                           (tag_x, tag_y), color)
        else:
            cv2.putText(frame,
                        '? {:.0f}%'.format(conf * 100),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 110), 2)

        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, 'FPS: {:.1f}'.format(fps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('AR Object Recognition  (Q=quit  S=screenshot)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('ar_screenshot.png', frame)
            print('Screenshot saved.')

    cap.release()
    cv2.destroyAllWindows()


def main(argv):
    args = parse_args()
    run_live(args)


if __name__ == '__main__':
    main(sys.argv)
