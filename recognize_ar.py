# Sangeeth Deleep Menon
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
# The FPS is shown in the top-left corner.
#
# Usage:
#   python recognize_ar.py                          # uses object_model.pth
#   python recognize_ar.py --model object_model.pth
#   python recognize_ar.py --conf 0.6 --roi 0.5
#
# Controls:
#   Q - quit
#   S - save screenshot to ar_screenshot.png

import sys
import argparse
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import ObjectCNN, ObjectViT, IMG_SIZE


# One distinct BGR colour per class (cycles if more than 8 classes)
_CLASS_COLORS = [
    (  0, 220,   0),   # green
    (  0, 140, 255),   # orange
    (200,   0, 200),   # magenta
    (255, 200,   0),   # cyan-blue
    (128,   0, 255),   # purple
    (  0, 200, 200),   # yellow-green
    (255,  80,  80),   # blue
    (  0, 200, 255),   # gold
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Real-time AR object recognition')
    p.add_argument('--model',  default='object_model.pth',
                   help='Path to the trained model checkpoint')
    p.add_argument('--camera', type=int,   default=0,
                   help='Camera device index')
    p.add_argument('--conf',   type=float, default=0.50,
                   help='Minimum confidence threshold (0-1) to show AR overlay')
    p.add_argument('--roi',    type=float, default=0.55,
                   help='ROI size as fraction of the shorter image dimension')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes    = checkpoint['classes']
    arch       = checkpoint.get('arch', 'cnn')
    img_size   = checkpoint.get('img_size', IMG_SIZE)

    model = ObjectViT(len(classes)) if arch == 'vit' else ObjectCNN(len(classes))
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
    """
    Estimates a pinhole camera intrinsic matrix from image dimensions.
    Focal length is approximated as 85% of the longer image dimension,
    which is a reasonable prior for typical webcam fields of view.
    """
    f  = max(w, h) * 0.85
    return np.array([[f,   0,  w / 2.0],
                     [0,   f,  h / 2.0],
                     [0,   0,  1.0    ]], dtype=np.float64)


# ---------------------------------------------------------------------------
# AR rendering
# ---------------------------------------------------------------------------

def project_3d_box(frame, roi_box, camera_matrix, color, thickness=2):
    """
    Draws a 3D wireframe box anchored to the detected ROI.

    The object is assumed to lie at a fixed depth of Z = 0.5 m.  Its 3D
    width and height are estimated by back-projecting the pixel ROI at that
    depth; the box depth is set to half the smaller of those two dimensions.
    cv2.projectPoints is used to map all eight corners back to pixel coords.

    Returns the projected 2D corner array (shape 8x2) so the caller can
    position the label tag above the box.
    """
    x1, y1, x2, y2 = roi_box
    cx_roi = (x1 + x2) / 2.0
    cy_roi = (y1 + y2) / 2.0

    f     = camera_matrix[0, 0]
    cx_c  = camera_matrix[0, 2]
    cy_c  = camera_matrix[1, 2]
    Z     = 0.5   # assumed depth in metres

    # Back-project ROI pixel size to metric dimensions at depth Z
    W = (x2 - x1) * Z / f
    H = (y2 - y1) * Z / f
    D = min(W, H) * 0.45   # box depth (slightly shallower than wide/tall)

    # 3D centre of the object (unproject ROI centre)
    X = (cx_roi - cx_c) * Z / f
    Y = (cy_roi - cy_c) * Z / f

    # 8 corners: indices 0-3 = front face, 4-7 = back face
    corners = np.array([
        [X - W/2, Y - H/2, Z - D/2],   # 0 front top-left
        [X + W/2, Y - H/2, Z - D/2],   # 1 front top-right
        [X + W/2, Y + H/2, Z - D/2],   # 2 front bot-right
        [X - W/2, Y + H/2, Z - D/2],   # 3 front bot-left
        [X - W/2, Y - H/2, Z + D/2],   # 4 back  top-left
        [X + W/2, Y - H/2, Z + D/2],   # 5 back  top-right
        [X + W/2, Y + H/2, Z + D/2],   # 6 back  bot-right
        [X - W/2, Y + H/2, Z + D/2],   # 7 back  bot-left
    ], dtype=np.float32)

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)

    pts, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, dist)
    pts    = pts.reshape(-1, 2).astype(int)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # front face
        (4, 5), (5, 6), (6, 7), (7, 4),   # back face
        (0, 4), (1, 5), (2, 6), (3, 7),   # connecting pillars
    ]
    for i, j in edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)

    return pts   # caller uses these to position the label


def draw_label_tag(frame, text, anchor_px, color, bg_alpha=0.72):
    """
    Draws a semi-transparent filled rectangle with dark text centred at
    anchor_px.  anchor_px is the (x, y) pixel of the bottom-centre of
    the tag (i.e. the tag floats upward from that point).
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

    # Clamp to frame bounds
    fh, fw = frame.shape[:2]
    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(fw, x1); y1 = min(fh, y1)

    # Semi-transparent coloured background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, cv2.FILLED)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)

    # Dark text on top
    tx = ax - tw // 2
    ty = ay - baseline - pad
    cv2.putText(frame, text, (tx, ty), font, font_scale, (10, 10, 10), thickness)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_roi(model, transform, roi_bgr, device):
    """Returns (class_index, confidence) for the given BGR ROI image."""
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(Image.fromarray(roi_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.exp(model(tensor))
    idx  = probs.argmax(1).item()
    conf = probs[0, idx].item()
    return idx, conf


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

def run_live(args):
    device = torch.device('mps'  if torch.backends.mps.is_available()  else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model, classes, img_size = load_model(args.model, device)
    transform = build_transform(img_size)
    print('Model loaded. Classes:', classes)
    print('Confidence threshold: {:.0f}%'.format(args.conf * 100))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Error: cannot open camera', args.camera)
        return

    prev_time  = time.time()
    fps        = 0.0

    print('\nAR recognition running.  Q = quit   S = screenshot\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        roi_box       = get_roi_box(h, w, args.roi)
        camera_matrix = build_camera_matrix(w, h)
        x1, y1, x2, y2 = roi_box

        # Classify the centre ROI
        roi      = frame[y1:y2, x1:x2]
        cls_idx, conf = classify_roi(model, transform, roi, device)
        class_name    = classes[cls_idx]
        color         = _CLASS_COLORS[cls_idx % len(_CLASS_COLORS)]

        # Always draw the 2D ROI border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if conf >= args.conf:
            # 3D wireframe box
            corner_pts = project_3d_box(frame, roi_box, camera_matrix, color)

            # Float the label tag above the top edge of the projected box
            top_pts    = corner_pts[:4]          # front face corners (0-3)
            tag_x      = int(top_pts[:, 0].mean())
            tag_y      = int(top_pts[:, 1].min()) - 8
            tag_y      = max(30, tag_y)           # keep inside frame
            label_text = '{} {:.0f}%'.format(class_name, conf * 100)
            draw_label_tag(frame, label_text, (tag_x, tag_y), color)
        else:
            # Below threshold: show greyed-out uncertain label
            cv2.putText(frame,
                        '? {:.0f}%'.format(conf * 100),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 110), 2)

        # FPS (exponential moving average)
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
            print('Screenshot saved to ar_screenshot.png')

    cap.release()
    cv2.destroyAllWindows()


def main(argv):
    args = parse_args()
    run_live(args)


if __name__ == '__main__':
    main(sys.argv)
