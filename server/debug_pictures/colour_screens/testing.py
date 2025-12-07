import os, re
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
cube_path = ROOT / "edge" / "data" / "scan_07December_10:14:44" / "cube_ZXnm.npz"
# Last inn .npz-fila
npz = np.load(cube_path)
cube = npz['cube']  # Antatt form (Z, X, Y, W)


def plot_row_spectrum(image, row, wavelengths_nm=None,
                      wl_min=400.0, wl_max=800.0, show=True):
    """
    Plott spekteret (intensitet vs bølgelengde) for én rad i et dispersjonsbilde.

    Parameters
    ----------
    image : np.ndarray
        2D-array (H, W) med gråtoneverdier fra kameraet.
    row : int
        Raden i bildet vi skal hente spekter fra (0 = øverste rad).
    wavelengths_nm : np.ndarray or None, optional
        Valgfri array med bølgelengder (W,). Hvis None brukes
        lineær mapping mellom wl_min og wl_max.
    wl_min : float
        Minimum bølgelengde i nm (brukes kun hvis wavelengths_nm=None).
    wl_max : float
        Maksimum bølgelengde i nm (brukes kun hvis wavelengths_nm=None).
    show : bool
        Hvis True vises plottet. Funksjonen returnerer alltid
        (wavelengths_nm, intensities).

    Returns
    -------
    wavelengths_nm : np.ndarray
        Bølgelengdeaksen (W,).
    intensities : np.ndarray
        Intensitetsverdier for valgt rad (W,).
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(f"image må være 2D (H, W), men er {image.shape}")

    h, w = image.shape

    if not (0 <= row < h):
        raise ValueError(f"row={row} er utenfor bildehøyden 0–{h-1}")

    # Intensiteter langs X for valgt rad
    intensities = image[row, :].astype(float)

    # Lag bølgelengdeaksen hvis den ikke er gitt
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(wl_min, wl_max, w)

    if wavelengths_nm.shape[0] != w:
        raise ValueError(
            f"wavelengths_nm må ha lengde {w}, men har {wavelengths_nm.shape[0]}"
        )

    if show:
        plt.figure()
        plt.plot(wavelengths_nm, intensities)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity [a.u.]")
        plt.title(f"Spectrum for row {row})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return wavelengths_nm, intensities



#Read image
image = cv2.imread("redscreen.png", cv2.IMREAD_GRAYSCALE)

#Flip image horizontally
#image = np.fliplr(image)   

#Choose row
row = int(image.shape[0] // 2*(1/3))

wavelengths_nm, intensities = plot_row_spectrum(image, row)
max_intensity = np.max(intensities)

print(f"Max intensity at red: {max_intensity}")
print(f"Corresponding wavelength at red: {wavelengths_nm[np.argmax(intensities)]} nm")