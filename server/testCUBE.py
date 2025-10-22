import os, re
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

#WORKS??? 
# ----------------- CONFIG -----------------
START_NM      = 400.0
END_NM        = 800.0
ROI_Y0, ROI_Y1 = 200, 400         # Region of interest, rows: (200:400)
NUM_BANDS = 968
# ------------------------------------------

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

# ---------- Collect filenames grouped by (Z, then X) ----------
pat = re.compile(r"X(\d+)[_-]Z(\d+)\.png", re.IGNORECASE)

rows = defaultdict(list)  # z -> list[(x, path)]
for f in os.listdir(scan_folder):
    if not (f.lower().endswith(".png") and f.upper().startswith("X")):
        continue
    m = pat.search(f)
    if not m:
        continue
    x10 = int(m.group(1))
    z10 = int(m.group(2))
    rows[z10].append((x10, os.path.join(scan_folder, f)))

if not rows:
    raise RuntimeError("No X*_Z*.png images found.")

Zs = sorted(rows.keys())  # Z rows (top→bottom)
Xc = None
H = W = None


# ---- Spectrum from one frame (mean over ROI rows) ----
def spectrum_from_frame(img):
    roi = img[ROI_Y0:ROI_Y1, :]         # (rows, W)
    return roi.mean(axis=0).astype(np.float32)  # (W,)


# ---- Optional: a small wrapper so cube[z,x,500] uses nm directly ----
class CubeNM:
    def __init__(self, data, wavs_nm):
        """
        data: ndarray (Z, X, 401) where axis=2 corresponds to wavs_nm=400..800
        wavs_nm: ndarray (401,) with integer nanometers: 400..800
        """
        self.data = data
        self.wavs_nm = wavs_nm.astype(np.float32)
        self.min_nm = float(wavs_nm[0])
        self.max_nm = float(wavs_nm[-1])

    def __getitem__(self, idx):
        # Support cube[z, x, nm] with nm in [400..800]
        if isinstance(idx, tuple) and len(idx) == 3 and isinstance(idx[2],(int, float, np.integer, np.floating)):
            z, x, nm = idx
            nm = float(nm)
            if nm < self.min_nm or nm > self.max_nm:
                raise IndexError(f"nm index {nm} out of range [{self.min_nm}, {self.max_nm}]")
            k = int(np.searchsorted(self.wavs_nm, nm, side='left'))
            if k == len(self.wavs_nm):
                k -= 1
            elif k > 0 and abs(self.wavs_nm[k] - nm) > abs(self.wavs_nm[k-1] - nm):
                k -= 1
            return self.data[z, x, k]
        return self.data[idx]

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data  # raw ndarray if you need it

# ---- Pass 1: read first frame to get H, W and wav grids; count X per row ----
first_z = Zs[0]
first_x_sorted = sorted(rows[first_z], key=lambda t: t[0])
Xc = len(first_x_sorted)

sample_img = cv2.imread(first_x_sorted[0][1], cv2.IMREAD_GRAYSCALE)
if sample_img is None:
    raise RuntimeError(f"Could not read {first_x_sorted[0][1]}")
H, W = sample_img.shape
print(f"Frame size: H={H}, W={W}")

wavs = np.linspace(START_NM, END_NM, W, dtype=np.float32)  # lengde = W (f.eks. 968)
print(f"Spectral grids: raw={len(wavs)} bands from {START_NM}nm to {END_NM}nm")

Zc = len(Zs)
cube_nm = np.zeros((Zc, Xc, len(wavs)), dtype=np.float32)  # (Z, X, 401)

# ---- Pass 2: fill the cube (Z, X, nm) ----
for zi, z in enumerate(Zs):
    x_paths = sorted(rows[z], key=lambda t: t[0])  # X ascending within this Z row
    if len(x_paths) != Xc:
        print(f"⚠️ Row Z={z} has {len(x_paths)} X positions (expected {Xc}).")
    for xi, (_x, path) in enumerate(x_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read {path}")
        if img.shape != (H, W):
            raise RuntimeError(f"Inconsistent frame size at {path}: {img.shape} vs {(H, W)}")

        spec_raw = spectrum_from_frame(img)               # (W,)
        spec_nm = spec_raw.astype(np.float32)           #Nødvendig? 
        cube_nm[zi, xi, :] = spec_nm

print("✅ Built cube_nm with shape (Z, X, nm):", cube_nm.shape)

# ---- Wrap so you can do cube[Z, X, 500] directly ----
cube = CubeNM(cube_nm, wavs)

def visualise_spectrum_at(cube, z, x):
    spec = cube[z, x, :]  # (nm,)
    print(f"Cube shape: {cube.shape}")
    print(f"Spectrum at Z={z}, X={x}:")
    axis = cube.wavs_nm
    plt.plot(axis, spec, label=f'Spectrum at Z={z}, X={x}')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()



# ---- Save to disk ----
npz_path = os.path.join(scan_folder, "cube_ZXnm.npz")
np.savez_compressed(npz_path, cube=cube_nm,      # raw ndarray (Z, X, 401)
                    wavs_nm=wavs,   # integer nm grid: 400..800
                    Zs=np.array(Zs, dtype=np.int32))
print("Saved:", npz_path)


def visualise_wavelength_slice(cube, wavelength_nm, out_path):
    """
    Visualize a single wavelength slice from the cube and save as PNG.

    Parameters:
        cube: CubeNM object
        wavelength_nm: Wavelength in nanometers to visualize
        out_path: Output file path for the PNG image
    """
    # Find the closest wavelength index
    k = int(np.searchsorted(cube.wavs_nm, wavelength_nm, side='left'))
    if k == len(cube.wavs_nm):
        k -= 1
    elif k > 0 and abs(cube.wavs_nm[k] - wavelength_nm) > abs(cube.wavs_nm[k-1] - wavelength_nm):
        k -= 1
    matched_nm = float(cube.wavs_nm[k])     #In case they are not the same 

    # Extract the 2D slice at this wavelength
    img = cube[:, :, matched_nm]  # (Z, X)

    # Normalize for visualization
    lo, hi = np.percentile(img, (1, 99))
    img_n = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_n, cmap="gray", aspect='auto')
    plt.xlabel("X Position")
    plt.ylabel("Z Position")
    plt.title(f"Slice of cube at wavelength: {cube.wavs_nm[k]:.1f} nm")
    plt.colorbar(label="Normalized Intensity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Wavelength {cube.wavs_nm[k]:.1f} nm -> {out_path}")
    

visualise_wavelength_slice(cube, 401, os.path.join(scan_folder, "wavelength_400nm.png"))

visualise_spectrum_at(cube, 35, 3)