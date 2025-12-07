import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
#  Calculation of NIR and RED indices
# ----------------------------------------------------------------
def calculate_nir_red_indices(cube: np.ndarray, wavelengths: np.ndarray) -> tuple:
    """
    Determine the index positions of the RED and NIR wavelengths in the hyperspectral cube.

    Parameters:
        cube: np.ndarray of shape (H, W, B)
        wavelengths: np.ndarray of shape (B,), containing the wavelength for each band

    Returns:
        A tuple (red_idx, nir_idx) with the indices closest to 660 nm and 800 nm.
    """

    if wavelengths.shape[0] != cube.shape[3]:
        raise ValueError("Wavelength array must have same length as number of bands in cube.")

    target_red = 660
    target_nir = 800

    red_idx = np.argmin(np.abs(wavelengths - target_red))
    nir_idx = np.argmin(np.abs(wavelengths - target_nir))

    print(f"Number of bands: {cube.shape[3]}")
    print(f"RED (≈660 nm) found at index: {red_idx}, wavelength={wavelengths[red_idx]:.1f} nm")
    print(f"NIR (≈800 nm) found at index: {nir_idx}, wavelength={wavelengths[nir_idx]:.1f} nm")

    return red_idx, nir_idx



def calculate_ndvi(cube: np.ndarray, red_band_idx: int, nir_band_idx: int) -> np.ndarray:
    """
    Compute an NDVI image from a hyperspectral cube using RED and NIR band indices.

    Parameters:
        cube: np.ndarray of shape (H, W, B)
        red_band_idx: index of the RED band
        nir_band_idx: index of the NIR band

    Returns:
        A float32 NDVI image with shape (H, W).
    """

    red = cube[:, :, :, red_band_idx].astype(np.float32)
    nir = cube[:, :, :, nir_band_idx].astype(np.float32)

    numerator = nir - red
    denominator = nir + red

    ndvi = np.full_like(red, np.nan, dtype=np.float32)
    mask = np.abs(denominator) < 1e-6

    np.divide(numerator, denominator, out=ndvi, where=~mask)
    ndvi[mask] = np.nan

    print(
        f"NDVI mean value: {np.nanmean(ndvi):.4f} "
        f"(RED mean={np.nanmean(red):.2f}, NIR mean={np.nanmean(nir):.2f})"
    )

    return ndvi