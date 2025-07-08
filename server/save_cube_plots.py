import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from Processing.SpectralTools import calculate_ndvi

# Hard paths
DATA_DIR = "/home/kybfarm/kybfarm/server/homeassistant/config/HSI/scanner_data"
OUTPUT_DIR = "/home/kybfarm/kybfarm/server/homeassistant/config/HSI/debug_pictures"

print(f"Scanner data: {DATA_DIR}")
print(f"Saving plots to: {OUTPUT_DIR}")

# Find latest scan
scan_folders = [f for f in os.listdir(DATA_DIR)
                if f.startswith("scan_") and os.path.isdir(os.path.join(DATA_DIR, f))]
scan_folders.sort(reverse=True)
latest_scan = scan_folders[0]
scan_folder = os.path.join(DATA_DIR, latest_scan)
cube_path = os.path.join(scan_folder, "hyperspectral_cube.npz")

data = np.load(cube_path)
cube = data["cube"]
print(f"Cube loaded: {cube.shape}")

# Grab CLI args
band_idx = int(sys.argv[1])
x = int(sys.argv[2])
y = int(sys.argv[3])
red_band = int(sys.argv[4])
nir_band = int(sys.argv[5])

# Save single band
def save_single_band(cube, band_idx, output_path):
    img = cube[:, :, band_idx]
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="gray")
    plt.title(f"Band {band_idx}")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved band image: {output_path}")

# Save pixel spectrum
def save_pixel_spectrum(cube, x, y, output_path):
    spectrum = cube[y, x, :]
    bands = np.arange(spectrum.shape[0])

    plt.figure(figsize=(8, 4))
    plt.plot(bands, spectrum)
    plt.xlabel("Band index")
    plt.ylabel("Intensity")
    plt.title(f"Pixel Spectrum ({x},{y})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved pixel spectrum: {output_path}")

# Save NDVI
def save_ndvi_map(cube, red_band, nir_band, output_path):
    ndvi = calculate_ndvi(cube, red_band, nir_band)
    plt.figure(figsize=(6, 5))
    plt.imshow(ndvi, cmap="RdYlGn")
    plt.title(f"NDVI Map (Red:{red_band} NIR:{nir_band})")
    plt.colorbar(label="NDVI")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved NDVI map: {output_path}")

# Do it
save_single_band(cube, band_idx, os.path.join(OUTPUT_DIR, "band.png"))
save_pixel_spectrum(cube, x, y, os.path.join(OUTPUT_DIR, "pixel_spectrum.png"))
save_ndvi_map(cube, red_band, nir_band, os.path.join(OUTPUT_DIR, "ndvi_map.png"))
