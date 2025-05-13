import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import numpy as np
from matplotlib import pyplot as plt
from Processing.SpectralTools import show_band, plot_pixel_spectrum, calculate_ndvi

# Set this to your scan directory
scan_folder = os.path.join(os.getcwd(), "data", "scan_30April_17:40:21")
cube_path = os.path.join(scan_folder, "hyperspectral_cube.npz")

# Load cube
print("Loading cube...")
data = np.load(cube_path)
cube = data["cube"]
print(f"Cube shape: {cube.shape} (H, W, Bands)")

# --- Show a single band ---
band_to_view = 240  # Adjust this index as needed
show_band(cube, band_idx=band_to_view)

# --- Plot spectrum of a pixel ---
x, y = 150, 300  # Adjust pixel coordinates
plot_pixel_spectrum(cube, x, y)

# --- NDVI (example: Red = band 60, NIR = band 120) ---
red_idx = 60
nir_idx = 120
ndvi = calculate_ndvi(cube, red_idx, nir_idx)

# Show NDVI map
plt.figure(figsize=(6, 5))
plt.imshow(ndvi, cmap='RdYlGn')
plt.title("NDVI Map")
plt.colorbar(label="NDVI")
plt.tight_layout()
plt.show()
