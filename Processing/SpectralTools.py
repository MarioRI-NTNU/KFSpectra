import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
#  NDVI Calculation
# ----------------------------------------------------------------
def calculate_ndvi(cube: np.ndarray, red_band_idx: int, nir_band_idx: int) -> np.ndarray:
    """
    Calculate NDVI from a hyperspectral cube.

    Parameters:
        cube: np.ndarray of shape (H, W, B)
        red_band_idx: index of the red band
        nir_band_idx: index of the NIR band

    Returns:
        NDVI image (H, W) as float32
    """
    red = cube[:, :, red_band_idx].astype(np.float32)
    nir = cube[:, :, nir_band_idx].astype(np.float32)
    bottom = nir + red
    bottom[bottom == 0] = 1e-5  # avoid division by zero
    ndvi = (nir - red) / bottom
    return ndvi

# ----------------------------------------------------------------
#  Plot spectrum of a single pixel
# ----------------------------------------------------------------
def plot_pixel_spectrum(cube: np.ndarray, x: int, y: int, wavelengths=None):
    """
    Plot the spectral signature of a pixel.

    Parameters:
        cube: np.ndarray of shape (H, W, B)
        x, y: coordinates of the pixel
        wavelengths: list or array of wavelengths (optional)
    """
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
    plt.show()

# ----------------------------------------------------------------
#  Visualize a single band
# ----------------------------------------------------------------
def show_band(cube: np.ndarray, band_idx: int, wavelength=None):
    """
    Display a grayscale image of a single band.

    Parameters:
        cube: np.ndarray of shape (H, W, B)
        band_idx: index of the band to show
        wavelength: optional label for the band
    """
    img = cube[:, :, band_idx]
    title = f"Band {band_idx}" + (f" ({wavelength} nm)" if wavelength else "")

    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()
