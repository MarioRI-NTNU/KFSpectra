import os, re
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

#WORKS??? 
# ----------------- CONFIG -----------------
START_NM      = 400.0
END_NM        = 800.0
#ROI_Y0, ROI_Y1 = 200, 400         # Region of interest, rows: (200:400)
NUM_BANDS = 968
# ------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "edge", "data")

print(f"Looking for scans in: {DATA_DIR}")

# Find latest scan folder - doesnt sort on month, only date
scan_folders = [f for f in os.listdir(DATA_DIR)
                if f.startswith("scan_") and os.path.isdir(os.path.join(DATA_DIR, f))]
if not scan_folders:
    raise FileNotFoundError(f"No scan folders found in: {DATA_DIR}")
scan_folders.sort(reverse=True)
latest_scan = scan_folders[0]
scan_folder = os.path.join(DATA_DIR, latest_scan)

#To choose a specific scan folder use this line instead: 
scan_folder = "/Users/hannahalse/KFSpectra/edge/data/scan_16November_13:58:13"
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
#def spectrum_from_frame(img):
#    roi = img[ROI_Y0:ROI_Y1, :]         # (rows, W)
#    return roi.mean(axis=0).astype(np.float32)  # (W,)


# ---- Optional: a small wrapper so cube[z,x,500] uses nm directly ----
class CubeNM:
    def __init__(self, data, wavs_nm):
        """
        data: ndarray (Z, X, Y, W) where axis=3 corresponds to wavelengths
        wavs_nm: ndarray (W,) with wavelengths in nm
        """
        self.data = data
        self.wavs_nm = wavs_nm.astype(np.float32)
        self.min_nm = float(wavs_nm[0])
        self.max_nm = float(wavs_nm[-1])

    def __getitem__(self, idx):
        # Support cube[z, x, y, nm] with nm in wavelength range
        if isinstance(idx, tuple) and len(idx) == 4 and isinstance(idx[3],(int, float, np.integer, np.floating)):
            z, x, y, nm = idx
            nm = float(nm)
            if nm < self.min_nm or nm > self.max_nm:
                raise IndexError(f"nm index {nm} out of range [{self.min_nm}, {self.max_nm}]")
            k = int(np.searchsorted(self.wavs_nm, nm, side='left'))
            if k == len(self.wavs_nm):
                k -= 1
            elif k > 0 and abs(self.wavs_nm[k] - nm) > abs(self.wavs_nm[k-1] - nm):
                k -= 1
            return self.data[z, x, y, k]
        return self.data[idx]

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

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
# ✨ NY STRUKTUR: (Z, X, Y, wavelength) - ett spektrum per rad
cube_nm = np.zeros((Zc, Xc, H, len(wavs)), dtype=np.float32)  # (Z, X, Y, W)

# ---- Pass 2: fill the cube (Z, X, Y, wavelength) ----
for zi, z in enumerate(Zs):
    #x_paths = sorted(rows[z], key=lambda t: t[0])
    x_paths = sorted(rows[z], key=lambda t: int(t[0]))
    if len(x_paths) != Xc:
        print(f"⚠️ Row Z={z} has {len(x_paths)} X positions (expected {Xc}).")
    
    for xi, (_x, path) in enumerate(x_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read {path}")
        if img.shape != (H, W):
            raise RuntimeError(f"Inconsistent frame size at {path}: {img.shape} vs {(H, W)}")

        # ✨ Lagre ALLE rader (Y-aksen) som separate spektra
        # img er (H, W) der hver rad er ett spektrum
        cube_nm[zi, xi, :, :] = img.astype(np.float32)  # (H, W) -> cube[zi, xi, :, :]

print("✅ Built cube_nm with shape (Z, X, Y, wavelength):", cube_nm.shape)
# Output: (38, 45, 480, 968) - 38 Z-pos, 45 X-pos, 480 rader, 968 wavelengths


# ---- Wrap so you can do cube[Z, X, 500] directly ----
cube = CubeNM(cube_nm, wavs)


# ---- Save to disk ----
npz_path = os.path.join(scan_folder, "cube_ZXnm.npz")
np.savez_compressed(npz_path, cube=cube_nm,      # raw ndarray (Z, X, 401)
                    wavs_nm=wavs,   # integer nm grid: 400..800
                    Zs=np.array(Zs, dtype=np.int32))
print("Saved:", npz_path)


# ---- Alternating symmetric X-shift (+5 / -5) per Z row ----
SHIFT_MAG = -5  # antall piksler halvpart av total differanse (total relat iv = 10)

def build_alternating_shifts(Zc, plus_first=True):
    """
    Returnerer array shape (Zc,) med +SHIFT_MAG og -SHIFT_MAG alternerende.
    plus_first=True gir [+d, -d, +d, -d, ...]
    """
    s = np.zeros(Zc, dtype=int)
    for zi in range(Zc):
        if plus_first:
            s[zi] = SHIFT_MAG if (zi % 2 == 0) else -SHIFT_MAG
        else:
            s[zi] = -SHIFT_MAG if (zi % 2 == 0) else SHIFT_MAG
    return s

def apply_integer_x_shifts(cube_nm, shifts, pad_value=np.nan):
    """
    cube_nm: (Z, X, Y, W), shifts: (Z,) heltall (positiv = flytt mot høyre).
    Return: shifted (med NaN i utfylte hull).
    """
    Zc, Xc, Yc, Wc = cube_nm.shape
    out = np.full_like(cube_nm, pad_value)
    for zi, s in enumerate(shifts):
        if s == 0:
            out[zi] = cube_nm[zi]
        elif s > 0:
            # flytt mot høyre: data havner lenger ut; venstre fylles med NaN
            out[zi, s:, :, :] = cube_nm[zi, :Xc - s, :, :]
        else:
            s2 = -s
            # flytt mot venstre
            out[zi, :Xc - s2, :, :] = cube_nm[zi, s2:, :, :]
    return out

def crop_valid_overlap(shifted):
    """
    Finn felles gyldig X-intervall (alle rader ikke-NaN).
    """
    Zc, Xc, Yc, Wc = shifted.shape
    mask = ~np.isnan(shifted[:, :, 0, 0])   # (Z, X)
    left = 0
    right = Xc
    # strengeste venstre grense
    for zi in range(Zc):
        row = mask[zi]
        if row.any():
            first = np.argmax(row)
            if first > left:
                left = first
    # strengeste høyre grense
    for zi in range(Zc):
        row = mask[zi]
        if row.any():
            last = Xc - np.argmax(row[::-1])
            if last < right:
                right = last
    xs = slice(left, right)
    return shifted[:, xs, :, :], xs

# Bygg altern ernede shifts (+5/-5). Juster plus_first hvis retning feil.
alt_shifts = build_alternating_shifts(Zc, plus_first=True)
print("Alternating X-shifts:", alt_shifts.tolist())

shifted_cube = apply_integer_x_shifts(cube_nm, alt_shifts, pad_value=np.nan)
cube_nm_sym, xslice = crop_valid_overlap(shifted_cube)
print("Sym-shifted & cropped cube shape:", cube_nm_sym.shape, "X slice:", xslice)

# Hvis du ønsker å fortsette videre med den korrigerte kuben:
cube_nm = cube_nm_sym
cube = CubeNM(cube_nm, wavs)


# ---- Visualiser spektrum fra én spesifikk rad ----
def visualise_spectrum_at(cube, z, x, y):
    """
    Visualize spectrum from a specific position and row.
    
    Parameters:
        cube: CubeNM object with shape (Z, X, Y, W)
        z: Z position
        x: X position  
        y: Y position (row number in the image)
    """
    spec = cube[z, x, y, :]  # ✅ (W,) - ett spektrum fra rad y
    print(f"Cube shape: {cube.shape}")
    print(f"Spectrum at Z={z}, X={x}, Y={y}:")
    
    plt.figure(figsize=(10, 6))
    plt.plot(cube.wavs_nm, spec, label=f'Z={z}, X={x}, Y-row={y}')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum at position (Z={z}, X={x}, Y={y})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
    

def reconstruct_rgb_image(cube):
    """
    Reconstruct an RGB image from the cube using specific wavelengths for R, G, B.

    Parameters:
        cube: CubeNM object with shape (Z, X, Y, W)
    """
# Define the target wavelengths for R, G, B channels
    r_wavelength = 620  # Red channel (620 nm)
    g_wavelength = 520  # Green channel (540 nm)
    b_wavelength = 460  # Blue channel (460 nm)

    # ✅ Find indices for these wavelengths
    r_idx = int(np.searchsorted(cube.wavs_nm, r_wavelength))
    g_idx = int(np.searchsorted(cube.wavs_nm, g_wavelength))
    b_idx = int(np.searchsorted(cube.wavs_nm, b_wavelength))
    
    print(f"RGB wavelengths: R={cube.wavs_nm[r_idx]:.1f}nm, G={cube.wavs_nm[g_idx]:.1f}nm, B={cube.wavs_nm[b_idx]:.1f}nm")

    # ✅ Extract channels using raw numpy array (not CubeNM wrapper)
    r_channel = cube.data[:, :, :, r_idx]  # (Z, X, Y)
    g_channel = cube.data[:, :, :, g_idx]  # (Z, X, Y)
    b_channel = cube.data[:, :, :, b_idx]  # (Z, X, Y)

    # Normalize each channel to 0-1 range
    def normalize_channel(ch):
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            return (ch - ch_min) / (ch_max - ch_min)
        return ch
    
    r_norm = normalize_channel(r_channel)
    g_norm = normalize_channel(g_channel)
    b_norm = normalize_channel(b_channel)

    # Stack the channels to create an RGB image
    rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)  # (Z, X, Y, 3)

    # Convert to uint8 for saving
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

    return rgb_image_uint8

#visualise_wavelength_slice(cube, 401, os.path.join(scan_folder, "wavelength_400nm.png"))
#y_middle = H // 2
#visualise_spectrum_at(cube, z=35, x=3, y=y_middle)

rgb_image = reconstruct_rgb_image(cube)
# Velg én Y-slice (f.eks. midten)
y_mid = H // 2
rgb_slice = rgb_image[:, :, y_mid, :]  # (Z, X, 3)

# Lagre
rgb_image_path = os.path.join(scan_folder, "reconstructed_rgb.png")
plt.imsave(rgb_image_path, rgb_slice)
print(f"✅ Saved RGB image to: {rgb_image_path}")

# Test for å sjekke spekter
z_sel = 20
x_sel = 10
y_sel = y_mid

spec = cube[z_sel, x_sel, y_sel, :]          # 1D (W,)
wavs_nm = cube.wavs_nm

# Lag figur
plt.figure(figsize=(8,4))
plt.plot(wavs_nm, spec)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title(f"Spectrum Z={z_sel} X={x_sel} Y={y_sel}")
plt.grid(alpha=0.3)
out_png = os.path.join(scan_folder, f"spectrum_Z{z_sel}_X{x_sel}_Y{y_sel}.png")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.close()
print(f"Saved spectrum plot: {out_png}")

# Lagre rå data (anbefales)
out_npy = os.path.join(scan_folder, f"spectrum_Z{z_sel}_X{x_sel}_Y{y_sel}.npy")
np.save(out_npy, spec)
print(f"Saved raw spectrum array: {out_npy}")

