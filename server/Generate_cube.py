import os
import sys
import numpy as np
import cv2

# ---------------------------
# CONFIG: Where scans live
# ---------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")  # Or your custom base path

print(f"Looking for scans in: {DATA_DIR}")

# ---------------------------
# Pick latest scan folder by name
# ---------------------------

scan_folders = [f for f in os.listdir(DATA_DIR)
                if f.startswith("scan_") and os.path.isdir(os.path.join(DATA_DIR, f))]

if not scan_folders:
    raise FileNotFoundError(f"No scan folders found in: {DATA_DIR}")

scan_folders.sort(reverse=True)
latest_scan = scan_folders[0]
scan_folder = os.path.join(DATA_DIR, latest_scan)

print(f"Using latest scan folder: {scan_folder}")

# ---------------------------
# Find images & sort by X (or by filename)
# ---------------------------

image_files = [f for f in os.listdir(scan_folder) if f.endswith(".png") and f.startswith("X")]
if not image_files:
    raise FileNotFoundError(f"No scan images found in: {scan_folder}")

# Sort by X value numerically
def get_x(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    xpart = [p for p in parts if p.startswith("X")][0]
    return int(xpart[1:])

image_files.sort(key=get_x)

print(f"Found {len(image_files)} images, sorted by X.")

# ---------------------------
# Load & stack
# ---------------------------

images = []
for fname in image_files:
    img_path = os.path.join(scan_folder, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"WARNING: Could not read image: {fname}")
        continue
    images.append(img)

if not images:
    raise RuntimeError("No valid images loaded!")

cube = np.stack(images, axis=2)  # (H, W, Bands)

print(f"Cube shape: {cube.shape} (Height, Width, Bands)")

# ---------------------------
# Save as NPZ
# ---------------------------

cube_path = os.path.join(scan_folder, "hyperspectral_cube.npz")
np.savez_compressed(cube_path, cube=cube)
print(f"Saved cube to: {cube_path}")
