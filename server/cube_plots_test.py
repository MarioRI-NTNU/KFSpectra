# save_cube_plots.py
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")                  # viktig på server
import matplotlib.pyplot as plt

#Importing functions from SpectralTools.py
from SpectralTools import (
    calculate_ndvi,        # bruker (H, W, B)
    plot_pixel_spectrum,   # viser spekter
    show_band,             # viser enkeltbånd
)

#Paths to input and output directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "edge", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "server", "debug_pictures")

#Finding the last scan and loading the cube
scan_folders = [f for f in os.listdir(DATA_DIR)
                if f.startswith("scan_") and os.path.isdir(os.path.join(DATA_DIR, f))]
if not scan_folders:
    raise RuntimeError("No scan folders found.")
scan_folders.sort(reverse=True)
scan_folder = os.path.join(DATA_DIR, scan_folders[0])
cube_path = os.path.join(scan_folder, "hyperspectral_cube.npz")

cube = np.load(cube_path)["cube"]      # din produksjon: (Y, W, X)
# Gjør om til (H, W, B) = (Y, X, Bands) som funksjonene dine forventer:
cube = np.transpose(cube, (0, 2, 1))
H, W, B = cube.shape
print(f"Cube: {cube.shape}  -> saving to {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
#Test:
print(f"Cube = {cube[277,472,620:630]}")

# --- args ---
if len(sys.argv) < 6:
    raise RuntimeError("Usage: python save_cube_plots.py BAND_IDX X Y RED_IDX NIR_IDX")
band_idx = int(sys.argv[1])
x        = int(sys.argv[2])
y        = int(sys.argv[3])
red_idx  = int(sys.argv[4])
nir_idx  = int(sys.argv[5])

# ----- wrappers som LAGRER png i stedet for å show() -----

def save_single_band(cube, band_idx, out_path, wavelength=None):
    # bruk show_band men fang figuren og lagre
    plt.figure(figsize=(6,5))
    img = cube[:, :, band_idx]
    plt.imshow(img, cmap="gray")
    title = f"Band {band_idx}" + (f" ({wavelength} nm)" if wavelength else "")
    plt.title(title)
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ band -> {out_path}")

def save_pixel_spectrum(cube, x, y, out_path, wavelengths=None):
    plt.figure(figsize=(8,4))
    spectrum = cube[y, x, :]
    bands = np.arange(spectrum.shape[0]) if wavelengths is None else wavelengths
    plt.plot(bands, spectrum, label=f"Pixel ({x},{y})")
    plt.xlabel("Wavelength (nm)" if wavelengths is not None else "Band index")
    plt.ylabel("Intensity")
    plt.title("Spectral Signature")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ spectrum -> {out_path}")

def save_ndvi_png(cube, red_idx, nir_idx, out_path):
    ndvi = calculate_ndvi(cube, red_idx, nir_idx)
    lo, hi = np.percentile(ndvi, [2, 98])
    ndvi_n = np.clip((ndvi - lo)/(hi - lo + 1e-6), 0, 1)
    plt.figure(figsize=(6,5))
    plt.imshow(ndvi_n, cmap="gray")
    plt.title(f"NDVI (red={red_idx}, nir={nir_idx})")
    plt.colorbar(label="relative NDVI"); plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ NDVI -> {out_path}")

# --- kjør ---
save_single_band(cube, band_idx, os.path.join(OUTPUT_DIR, f"band_{band_idx}.png"))
save_pixel_spectrum(cube, x, y, os.path.join(OUTPUT_DIR, f"spectrum_x{x}_y{y}.png"))
save_ndvi_png(cube, red_idx, nir_idx, os.path.join(OUTPUT_DIR, f"ndvi_r{red_idx}_n{nir_idx}.png"))

# (valgfritt) rask RGB-preview
wavs = np.linspace(400, 800, B)
r_i = int(np.argmin(np.abs(wavs-650)))
g_i = int(np.argmin(np.abs(wavs-550)))
b_i = int(np.argmin(np.abs(wavs-450)))
rgb = np.stack([cube[:,:,r_i], cube[:,:,g_i], cube[:,:,b_i]], axis=-1).astype(np.float32)
lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)
rgb = np.clip((rgb - lo)/(hi - lo + 1e-6), 0, 1)
plt.figure(figsize=(6,5)); plt.imshow(rgb); plt.axis("off"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "preview_rgb.png"), dpi=200); plt.close()
print("Done.")

