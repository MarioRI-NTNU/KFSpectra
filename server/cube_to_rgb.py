import sys, numpy as np
from PIL import Image

if len(sys.argv) < 2:
    print("bruk: python simple_cube_to_rgb.py <cube.npz>")
    raise SystemExit

data = np.load(sys.argv[1])
cube = data["cube"]                       # (H, W, B)
wavs = data["wavs_nm"] if "wavs_nm" in data.files else np.linspace(400, 800, cube.shape[2])

def idx(nm): return int(np.argmin(np.abs(wavs.astype(float) - float(nm))))

r = cube[:, :, idx(660.0)].astype(np.float64)
g = cube[:, :, idx(550.0)].astype(np.float64)
b = cube[:, :, idx(450.0)].astype(np.float64)

def norm01(x):
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)

rgb = np.stack([norm01(r), norm01(g), norm01(b)], axis=2)
rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)

Image.fromarray(rgb8).save("rgb.png")
print("lagret: rgb.png")
