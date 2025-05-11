import os
import argparse
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_image(image_path: str, padding: int = 0) -> np.ndarray:
    """
    Crop the main object in an image using Otsu's thresholding and morphological operations.

    Args:
        image_path: Path to the input image file.
        padding: Number of pixels to add around the bounding box.

    Returns:
        Cropped image as an RGB NumPy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, img.shape[1])
    y1 = min(y + h + padding, img.shape[0])

    cropped = img[y0:y1, x0:x1]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def process_directory(input_dir: str, output_dir: str, padding: int) -> None:
    """
    Crop all images in `input_dir` and save to `output_dir`.
    Creates `output_dir` if it doesn't exist.
    """
    if not os.path.isdir(input_dir):
        print(f"Warning: directory not found: {input_dir}, skipping.")
        return
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith('.'):
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        try:
            cropped = crop_image(src, padding)
            plt.imsave(dst, cropped)
        except Exception as e:
            print(f"Failed to process {src}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AutoCrop: batch-crop MRIs in Testing/ and Training/ folders."
    )
    parser.add_argument('-p', '--padding', type=int, default=0,
                        help="Extra pixels around each detected bounding box.")
    parser.add_argument('-i', '--input_root', type=str, default=None,
                        help="Explicit input root containing Testing/ and Training/. (e.g., brain_tumor_d)")
    parser.add_argument('-o', '--output_root', type=str, default=None,
                        help="Explicit output root for cropped folders (default: script folder)")
    args, _ = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = args.input_root or os.path.join(script_dir, 'brain_tumor_d')
    output_root = args.output_root or script_dir

    print(f"Reading from input root: {input_root}")
    print(f"Writing to output root: {output_root}")

    for split in ('Testing', 'Training'):
        in_split = os.path.join(input_root, split)
        out_split = os.path.join(output_root, f"{split}_cropped")
        if not os.path.isdir(in_split):
            print(f"Warning: {in_split} not found, skipping split.")
            continue
        os.makedirs(out_split, exist_ok=True)
        for class_name in sorted(os.listdir(in_split)):
            in_dir = os.path.join(in_split, class_name)
            if not os.path.isdir(in_dir) or class_name.startswith('.'):
                continue
            out_dir = os.path.join(out_split, class_name)
            process_directory(in_dir, out_dir, args.padding)


if __name__ == '__main__':
    main()
