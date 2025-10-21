import os
import numpy as np
import cv2
import re

# Paths
#DATA_DIR = "/home/kybfarm/kybfarm/server/homeassistant/config/HSI/scanner_data"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "edge", "data")

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

#---------------- Changes made from here ----------------
pat = re.compile(r"X(\d+)[_\-]Z(\d+)", re.IGNORECASE)

image_files = [f for f in os.listdir(scan_folder)
               if f.lower().endswith(".png") and f.upper().startswith("X")]

def zx_key(fname):
    m = pat.search(fname)
    if not m:  # legg ukjente til slutt, stabilt
        return (10**9, 10**9, fname)
    x = int(m.group(1))
    z = int(m.group(2))
    return (z, x, fname)   # ← Z først, så X

image_files.sort(key=zx_key)

images = []
for fname in image_files:
    img = cv2.imread(os.path.join(scan_folder, fname), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        images.append(img)

cube = np.stack(images, axis=2)  # (H, W_spektral, B=antall (X,Z)-posisjoner)

#----------------- To here -------------------

npz_path = os.path.join(scan_folder, "hyperspectral_cube.npz")
np.savez_compressed(npz_path, cube=cube)
print(f"Saved cube to: {npz_path}")
