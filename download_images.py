# Sangeeth Deleep Menon
# NUID: 002524579
# CS5330 Final Project - Web Image Downloader
# Spring 2026
#
# Downloads additional training images from Bing image search using icrawler.
# Images are saved alongside the webcam-captured images in data/<class_name>/
# so train.py sees them automatically.
#
# Usage:
#   python download_images.py                      # default 80 images per class
#   python download_images.py --per-class 120
#   python download_images.py --classes cup phone  # specific classes only
#
# Requirements:
#   pip install icrawler

import os
import sys
import argparse
from icrawler.builtin import BingImageCrawler

DEFAULT_CLASSES = ['cup', 'phone', 'book', 'keyboard', 'pen']
DEFAULT_DATA_DIR = 'data'

# More specific search queries per class to get cleaner results
SEARCH_QUERIES = {
    'cup':            'ceramic coffee mug cup object white background',
    'phone':          'smartphone mobile phone object on desk',
    'book':           'hardcover book object on table',
    'keyboard':       'computer keyboard object desk',
    'pen':            'ballpoint pen writing pen object',
    'ps5_controller': 'sony ps5 dualsense controller gaming',
    'glasses':        'eyeglasses spectacles reading glasses object',
    'headphones':     'over ear headphones audio headset object desk',
    'laptop':         'laptop computer open desk side view',
    'tablet':         'tablet ipad device screen object desk',
}


def parse_args():
    p = argparse.ArgumentParser(description='Download training images from Bing')
    p.add_argument('--classes',   nargs='+', default=DEFAULT_CLASSES)
    p.add_argument('--data',      default=DEFAULT_DATA_DIR)
    p.add_argument('--per-class', type=int,  default=80,
                   help='Number of images to download per class')
    return p.parse_args()


def count_existing(data_dir, cls):
    cls_dir = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_dir):
        return 0
    return len([f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def download_class(cls, data_dir, n_images):
    cls_dir = os.path.join(data_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)

    # Find the next available file number to avoid overwriting webcam images
    existing = count_existing(data_dir, cls)
    query = SEARCH_QUERIES.get(cls, '{} object'.format(cls))

    print('  Downloading {} images for "{}" (query: "{}")'.format(
        n_images, cls, query))

    crawler = BingImageCrawler(
        storage={'root_dir': cls_dir},
        downloader_threads=4,
        log_level='ERROR',   # suppress verbose icrawler output
    )

    # icrawler names files 000001.jpg etc.; we use a temp sub-dir then rename
    # to keep consistent numbering with webcam images
    tmp_dir = os.path.join(data_dir, '_tmp_' + cls)
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_crawler = BingImageCrawler(
        storage={'root_dir': tmp_dir},
        downloader_threads=4,
        log_level='ERROR',
    )
    tmp_crawler.crawl(keyword=query, max_num=n_images,
                      file_idx_offset=0)

    # Rename and move into the class directory with sequential numbering
    downloaded = sorted(f for f in os.listdir(tmp_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    moved = 0
    for i, fname in enumerate(downloaded):
        new_name = '{:04d}.jpg'.format(existing + i + 1)
        src = os.path.join(tmp_dir, fname)
        dst = os.path.join(cls_dir, new_name)
        os.rename(src, dst)
        moved += 1

    # Clean up temp dir
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return moved


def main(argv):
    args = parse_args()

    print('Web image download starting.')
    print('Saving to:', os.path.abspath(args.data))
    print('Classes:  ', args.classes)
    print('Target:    {} new images per class\n'.format(args.per_class))

    totals = {}
    for cls in args.classes:
        before = count_existing(args.data, cls)
        n = download_class(cls, args.data, args.per_class)
        after = count_existing(args.data, cls)
        totals[cls] = after
        print('  {} -> {} images downloaded, {} total\n'.format(cls, n, after))

    print('--- Download summary ---')
    grand_total = 0
    for cls in args.classes:
        print('  {:15s}  {} images'.format(cls, totals[cls]))
        grand_total += totals[cls]
    print('  {:15s}  {} images total'.format('', grand_total))


if __name__ == '__main__':
    main(sys.argv)
