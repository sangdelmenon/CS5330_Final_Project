# Sangeeth Deleep Menon
# CS5330 Final Project - Webcam Data Collection Tool
# Spring 2026
#
# Opens the webcam and lets you capture training images for each object class.
# Images are saved to  data/<class_name>/<nnnn>.jpg
#
# Usage:
#   python collect_data.py
#   python collect_data.py --classes cup phone book keyboard pen
#   python collect_data.py --classes mug laptop pen --camera 1
#
# Controls:
#   SPACE  - capture one frame for the active class
#   A      - toggle auto-capture (one frame every AUTO_INTERVAL frames)
#   0-9    - switch active class by index
#   Q      - quit and print collection summary

import sys
import os
import argparse
import cv2

DEFAULT_CLASSES  = ['cup', 'phone', 'book', 'keyboard', 'pen']
DEFAULT_DATA_DIR = 'data'
ROI_FRACTION     = 0.55    # ROI square is this fraction of the shorter image side
AUTO_INTERVAL    = 12      # frames between auto-capture shots


def parse_args():
    p = argparse.ArgumentParser(description='Collect training images from webcam')
    p.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES,
                   help='Space-separated list of object class names')
    p.add_argument('--camera', type=int, default=0, help='Camera device index')
    p.add_argument('--data',   default=DEFAULT_DATA_DIR,
                   help='Root directory where class sub-folders are created')
    return p.parse_args()


def ensure_dirs(data_dir, classes):
    for cls in classes:
        os.makedirs(os.path.join(data_dir, cls), exist_ok=True)


def count_images(data_dir, cls):
    cls_dir = os.path.join(data_dir, cls)
    return len([f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def get_roi_box(h, w, fraction=ROI_FRACTION):
    """Returns (x1, y1, x2, y2) for a centred square ROI."""
    size = int(min(h, w) * fraction)
    cx, cy = w // 2, h // 2
    return cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2


def save_roi(roi, data_dir, class_name, existing_count):
    path = os.path.join(data_dir, class_name, '{:04d}.jpg'.format(existing_count + 1))
    cv2.imwrite(path, roi)
    return path


def draw_ui(frame, classes, active_idx, data_dir, auto_capture, roi_box):
    x1, y1, x2, y2 = roi_box
    h = frame.shape[0]

    # ROI rectangle – green normally, orange in auto-capture mode
    box_color = (0, 140, 255) if auto_capture else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Class list on the left side
    for i, cls in enumerate(classes):
        cnt    = count_images(data_dir, cls)
        marker = '>' if i == active_idx else ' '
        text   = '{} [{}] {}  ({})'.format(marker, i, cls, cnt)
        color  = (0, 255, 0) if i == active_idx else (160, 160, 160)
        cv2.putText(frame, text, (10, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Active class banner above the ROI box
    active_cls = classes[active_idx]
    banner = 'Capturing: {}  ({} imgs)'.format(
        active_cls, count_images(data_dir, active_cls))
    cv2.putText(frame, banner, (x1, y1 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

    # Controls hint at the bottom
    hint = 'SPACE=capture  A=auto[{}]  0-{} switch class  Q=quit'.format(
        'ON' if auto_capture else 'off', min(9, len(classes) - 1))
    cv2.putText(frame, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


def run_collection(classes, data_dir, camera_idx):
    ensure_dirs(data_dir, classes)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print('Error: cannot open camera {}'.format(camera_idx))
        return

    active_idx   = 0
    auto_capture = False
    frame_num    = 0

    print('Data collection started.')
    print('Classes:', classes)
    print('Saving to:', os.path.abspath(data_dir))

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame.')
            break

        h, w = frame.shape[:2]
        roi_box = get_roi_box(h, w)
        x1, y1, x2, y2 = roi_box

        display = frame.copy()
        draw_ui(display, classes, active_idx, data_dir, auto_capture, roi_box)

        # Auto-capture: save one frame every AUTO_INTERVAL frames
        if auto_capture and frame_num % AUTO_INTERVAL == 0:
            roi = frame[y1:y2, x1:x2]
            n   = count_images(data_dir, classes[active_idx])
            save_roi(roi, data_dir, classes[active_idx], n)
            # Brief white flash to show a frame was captured
            display[y1:y2, x1:x2] = (
                display[y1:y2, x1:x2].astype('float32') * 0.5 + 127
            ).astype('uint8')

        frame_num += 1
        cv2.imshow('Data Collection  (Q=quit)', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            roi  = frame[y1:y2, x1:x2]
            n    = count_images(data_dir, classes[active_idx])
            path = save_roi(roi, data_dir, classes[active_idx], n)
            print('Saved: {}'.format(path))
        elif key == ord('a'):
            auto_capture = not auto_capture
            print('Auto-capture:', 'ON' if auto_capture else 'OFF')
        elif ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < len(classes):
                active_idx = idx
                print('Active class -> {} ({})'.format(
                    classes[active_idx], active_idx))

    cap.release()
    cv2.destroyAllWindows()

    print('\n--- Collection summary ---')
    total = 0
    for cls in classes:
        n = count_images(data_dir, cls)
        total += n
        print('  {:15s}  {} images'.format(cls, n))
    print('  {:15s}  {} images total'.format('', total))


def main(argv):
    args = parse_args()
    run_collection(args.classes, args.data, args.camera)


if __name__ == '__main__':
    main(sys.argv)
