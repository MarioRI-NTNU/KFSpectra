import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-GUI backend
import matplotlib.pyplot as plt

from Processing.SpectralTools import calculate_ndvi

# Hard paths
DATA_DIR = "/home/kybfarm/kybfarm/server/homeassistant/config/HSI/scanner_data"
OUTPUT_DIR = "/home/kybfarm/kybfarm/server/homeassistant/config/HSI/debug_pictures"

print(f"Scanner data: {DATA_DIR}")
print(f"Saving PNGs to: {OUTPUT_DIR}")

# Find latest scan
scan_folders = [f for f in os.listdir(DATA_DIR)
                if f.startswith("scan_") and os.path.isdir(os.path.join(DATA_DIR, f))]
if not scan_folders:
    raise RuntimeError("No scan folders found.")
scan_folders.sort(reverse=True)
latest_scan = scan_folders[0]
scan_folder = os.path.join(DATA_DIR, latest_scan)
cube_path = os.path.join(scan_folder, "hyperspectral_cube.npz")

data = np.load(cube_path)
cube = data["cube"]
print(f"Cube loaded: {cube.shape}")

# Args
if len(sys.argv) < 6:
    raise RuntimeError("Usage: save_cube_plots.py BAND_IDX X Y RED_IDX NIR_IDX")

band_idx = int(sys.argv[1])
x = int(sys.argv[2])
y = int(sys.argv[3])
red_band = int(sys.argv[4])
nir_band = int(sys.argv[5])

# Single band image
def save_single_band(cube, band_idx, output_path):
    img = cube[:, :, band_idx]
    plt.imshow(img, cmap="gray")
    plt.title(f"Band {band_idx}")
    plt.colorbar(label="Intensity")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Band PNG saved: {output_path}")

# Pixel spectrum
def save_pixel_spectrum(cube, x, y, output_path):
    spectrum = cube[y, x, :]
    bands = np.arange(spectrum.shape[0])
    plt.plot(
