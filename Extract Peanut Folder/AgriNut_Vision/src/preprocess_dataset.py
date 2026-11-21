"""
preprocess_dataset.py  -- raw peanut images ko crop + resize karta hai
========================================================================
Input : dataset/good/*.jpg   dataset/bad/*.jpg   (raw full-box images)
Output: dataset_processed/good/*.jpg  dataset_processed/bad/*.jpg  (224x224, sirf peanut)

Yehi crop_peanut() function LIVE deployment mein bhi use hoga (bilkul same),
warna domain gap aa jayega.

Chalane ka tareeqa:  python preprocess_dataset.py
"""

import cv2
import numpy as np
import os
import glob

# ---- settings (box fixed hai, inko mat badlo jab tak zaroorat na ho) ----
IN_DIR  = "dataset"
OUT_DIR = "dataset_processed"
TOP, BOT = 0.24, 0.80      # wood top/bottom hatane ke liye belt strip
IMG = 224                  # final size (MobileNetV2 input)
MARGIN = 0.25              # peanut ke around extra jagah
MIN_AREA = 2000            # isse chhota blob = noise, peanut nahi
# --------------------------------------------------------------------------


def crop_peanut(im):
    """full image -> 224x224 crop sirf peanut ka. None agar peanut na mile."""
    h, w = im.shape[:2]
    y1, y2 = int(h * TOP), int(h * BOT)
    belt = im[y1:y2, :]
    gray = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((9, 9), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    cx, cy = x + bw / 2, y + bh / 2
    side = int(max(bw, bh) * (1 + MARGIN))
    bx1 = max(0, int(cx - side / 2)); by1 = max(0, int(cy - side / 2))
    bx2 = min(belt.shape[1], int(cx + side / 2)); by2 = min(belt.shape[0], int(cy + side / 2))
    crop = belt[by1:by2, bx1:bx2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG, IMG))


def process_folder(cls):
    in_path  = os.path.join(IN_DIR, cls)
    out_path = os.path.join(OUT_DIR, cls)
    os.makedirs(out_path, exist_ok=True)
    files = glob.glob(os.path.join(in_path, "*.jpg")) + glob.glob(os.path.join(in_path, "*.png"))
    ok, fail = 0, 0
    for i, f in enumerate(files):
        im = cv2.imread(f)
        if im is None:
            fail += 1; continue
        crop = crop_peanut(im)
        if crop is None:
            fail += 1
            print("   SKIP (peanut na mila):", os.path.basename(f))
            continue
        cv2.imwrite(os.path.join(out_path, f"{cls}_{ok:04d}.jpg"), crop)
        ok += 1
    print(f"[{cls}]  OK={ok}   skipped={fail}   (total={len(files)})")
    return ok, fail


if __name__ == "__main__":
    print("Preprocessing shuru...\n")
    g_ok, g_f = process_folder("good")
    b_ok, b_f = process_folder("bad")
    print(f"\nDONE. good={g_ok}  bad={b_ok}  -> folder: {OUT_DIR}/")
    if g_f or b_f:
        print(f"NOTE: {g_f+b_f} images skip hui (peanut detect nahi hua). "
              f"Agar zyada skip hui to mujhe batao.")
