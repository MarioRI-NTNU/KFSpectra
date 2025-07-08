import os
import numpy as np
import cv2

# Paths
DATA_DIR = "/home/kybfarm/kybfarm/server/homeassistant/config/HSI/scanner_data"

print(f"Looking for scans in: {DATA_DIR}")

# Find latest scan folder
scan_folders = [f for f in os.listdir(DATA_DIR)
                if f.startswith("scan_") and os.path.isdir(os.path.join(DATA_DIR, f))]
if not scan_folders:
    raise FileNotFoundError(f"No scan folders found in: {DATA_DIR}")

scan_folders.sort(reverse=True)
latest_scan = scan_folders[0]
scan_folder = os.path.join(DATA_DIR, latest_scan)

print(f"Using latest scan: {scan_folder}")

# Find images
image_files = [f for f in os.listdir(scan_folder) if f.endswith(".png") and f.startswith("X")]
image_files.sort()  # or sort by X number

images = []
for fname in image_files:
    img_path = os.path.join(scan_folder, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: could not read {fname}")
        continue
    images.append(img)

if not images:
    raise RuntimeError("No valid images loaded.")

cube = np.stack(images, axis=2)  # (H, W, Bands)

npz_path = os.path.join(scan_folder, "hyperspectral_cube.npz")
np.savez_compressed(npz_path, cube=cube)
print(f"Saved cube to: {npz_path}")
