import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import cv2
import numpy as np
from pyueye import ueye
from time import sleep
from acquisition.Camera_Control import Camera 

# --- Parameters ---
bin_size_x = 8
start_wavelength = 400
end_wavelength = 800
calibration_file = "calibration.json"

# --- Helpers ---
def crop_useful_area(img):
    y_start = 296
    y_end = 845
    x_start = 0
    x_end = 1936
    return img[y_start:y_end, x_start:x_end]

def spectral_binning(image_row, bin_size=8):
    return np.array([
        np.mean(image_row[i:i+bin_size]) for i in range(0, len(image_row), bin_size)
    ])

def capture_color_frame(camera, label):
    input(f"\n➡️ Please place a solid {label.upper()} object in view and press ENTER...")
    sleep(0.5)
    frame = camera.capture_frame()
    cropped = crop_useful_area(frame)
    return cropped

def compute_average_spectrum(cropped):
    height = cropped.shape[0]
    binned_rows = np.array([
        spectral_binning(cropped[y, :], bin_size=bin_size_x) for y in range(height)
    ])
    return binned_rows.mean(axis=0)

def find_peak_band(wavelengths, spectrum, target_range):
    mask = (wavelengths >= target_range[0]) & (wavelengths <= target_range[1])
    sub = spectrum[mask]
    peak_idx = np.argmax(sub)
    return np.where(mask)[0][peak_idx]

# --- Main Calibration Logic ---
def run_calibration():
    print("\n🎛 Starting hyperspectral RGB band calibration...\n")
    cam = Camera()
    cam.connect()

    try:
        spectra = {}
        for color in ["red", "green", "blue"]:
            cropped = capture_color_frame(cam, color)
            avg_spectrum = compute_average_spectrum(cropped)
            spectra[color] = avg_spectrum

        wavelengths = np.linspace(start_wavelength, end_wavelength, len(spectra["red"]))

        red_band = find_peak_band(wavelengths, spectra["red"], (600, 700))
        green_band = find_peak_band(wavelengths, spectra["green"], (510, 580))
        blue_band = find_peak_band(wavelengths, spectra["blue"], (420, 500))

        calib_data = {
            "red_band": int(red_band),
            "green_band": int(green_band),
            "blue_band": int(blue_band)
        }

        with open(calibration_file, "w") as f:
            json.dump(calib_data, f, indent=2)

        print("\n✅ Calibration complete!")
        print(f"Selected bands -> R: {red_band}, G: {green_band}, B: {blue_band}")
        print(f"Saved to {calibration_file}")

    finally:
        cam.disconnect()

if __name__ == "__main__":
    run_calibration()
