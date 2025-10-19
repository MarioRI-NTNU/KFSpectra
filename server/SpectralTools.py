import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
#  Calculation of NIR and RED indices
# ----------------------------------------------------------------
def calculate_nir_red_indices(cube:np.ndarray) -> tuple:
    """
    Calculate NIR and RED indices from a hyperspectral cube.

    Parameters:
        cube: np.ndarray of shape (H, W, B)

    Returns:
        A tuple containing the RED and NIR indices.
    """

    wavelengths = np.linspace(400, 800, cube.shape[2])  #Camera takes wavelengths from 400nm to 800nm 
    print(f"Number of bands: {cube.shape[2]}")
    red_idx = np.argmin(np.abs(wavelengths - 660))  # Approx 660nm for RED
    nir_idx = np.argmin(np.abs(wavelengths - 800))  # Approx 800nm for NIR
    print(f"Calculated RED index: {red_idx}, NIR index: {nir_idx}")
    return red_idx, nir_idx


# ----------------------------------------------------------------
#  NDVI Calculation - Maybe make it so that it returns only the ones that actually detects a plant? Not background? 
# ----------------------------------------------------------------
def calculate_ndvi(cube: np.ndarray, red_band_idx: int, nir_band_idx: int) -> np.ndarray:
    """
    Calculate NDVI from a hyperspectral cube.

    Parameters:
        cube: np.ndarray of shape (H, W, B)
        red_band_idx: index of the red band
        nir_band_idx: index of the NIR band

    Returns:
        NDVI image (H, W) as float32 (array)
    """
    ndvi = np.full(cube.shape[:2], np.nan, dtype=np.float32)
    red = cube[:, :, red_band_idx].astype(np.float32)
    nir = cube[:, :, nir_band_idx].astype(np.float32)
    num = nir - red
    den = nir + red
    small_val = 1e-6
    mask = np.abs(den) < small_val
    np.divide(num, den, out=ndvi, where=~mask)  #To avoid dividing by zero
    ndvi[mask] = np.nan
    mean = np.nanmean(ndvi)             #Ignores NaN values
    print(f"Mean NDVI calculates is: {mean}. Red mean: {np.nanmean(red)}, NIR mean: {np.nanmean(nir)}")
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
