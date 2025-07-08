import sys
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Processing.SpectralTools import calculate_ndvi

# ------------------------------------
# Load the latest cube (or pick a path)
# ------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")

scan_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("scan_") and os.path.isdir(os.path.join(DATA_DIR, f))]
scan_folders.sort(reverse=True)
latest_scan = scan_folders[0]

scan_folder = os.path.join(DATA_DIR, latest_scan)
cube_path = os.path.join(scan_folder, "hyperspectral_cube.npz")

print(f"Using scan: {scan_folder}")

data = np.load(cube_path)
cube = data["cube"]

print(f"Loaded cube shape: {cube.shape} (H, W, Bands)")

# ------------------------------------
# Functions that generate + save plots
# ------------------------------------

def save_single_band(cube, band_idx, output_path, wavelength=None):
    img = cube[:, :, band_idx]
    plt.figure(figsize=(6, 5))
    title = f"Band {band_idx}" + (f" ({wavelength} nm)" if wavelength else "")
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved single band to {output_path}")

def save_pixel_spectrum(cube, x, y, output_path, wavelengths=None):
    spectrum = cube[y, x, :]
    bands = np.arange(spectrum.shape[0]) if wavelengths is None else wavelengths

    plt.figure(figsize=(8, 4))
    plt.plot(bands, spectrum, label=f"Pixel ({x}, {y})")
    plt.xlabel("Wavelength (nm)" if wavelengths is not None else "Band index")
    plt.ylabel("Intensity")
    plt.title("Spectral Signature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved pixel spectrum to {output_path}")

def save_ndvi_map(cube, red_band_idx, nir_band_idx, output_path):
    ndvi = calculate_ndvi(cube, red_band_idx, nir_band_idx)

    plt.figure(figsize=(6, 5))
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.title("NDVI Map")
    plt.colorbar(label="NDVI")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved NDVI map to {output_path}")

# ------------------------------------
# Example usage
# ------------------------------------

if __name__ == "__main__":
    out_band = os.path.join(scan_folder, "band_240.png")
    out_spectrum = os.path.join(scan_folder, "pixel_spectrum_150_300.png")
    out_ndvi = os.path.join(scan_folder, "ndvi_map.png")

    save_single_band(cube, band_idx=240, output_path=out_band)
    save_pixel_spectrum(cube, x=150, y=300, output_path=out_spectrum)
    save_ndvi_map(cube, red_band_idx=60, nir_band_idx=120, output_path=out_ndvi)
