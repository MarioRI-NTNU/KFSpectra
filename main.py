import os
import cv2
import numpy as np
import json
from matplotlib import pyplot as plt

# --- Parameters ---
bin_size_x = 8       # 8 pixels per spectral band
start_wavelength = 400  # nm
end_wavelength = 800    # nm

def crop_useful_area(img):
    y_start = 296
    y_end = 845
    x_start = 0
    x_end = 1936
    cropped = img[y_start:y_end, x_start:x_end]
    return cropped

def spectral_binning(image_row, bin_size=8):
    return np.array([
        np.mean(image_row[i:i+bin_size]) for i in range(0, len(image_row), bin_size)
    ])

def load_calibrated_rgb_bands(calibration_path="calibration.json"):
    with open(calibration_path, "r") as f:
        calib = json.load(f)
    return calib["red_band"], calib["green_band"], calib["blue_band"]

def correct_perspective(image, scale_y=0.45):
    """
    Apply vertical scaling to correct aspect ratio distortion.
    Adjust scale_y based on visual feedback (e.g., 0.6 flattens the ellipse into a circle).
    """
    height, width = image.shape[:2]
    new_height = int(height * scale_y)
    corrected = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LINEAR)
    return corrected


def create_hyperspectral_cube(scan_folder):
    # Load all images
    images = []
    for filename in sorted(os.listdir(scan_folder)):
        if filename.endswith(".png") and filename.startswith("X"):
            img_path = os.path.join(scan_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                cropped = crop_useful_area(img)
                images.append(cropped)

    if not images:
        print("No images found!")
        return None

    # Assume all cropped images are same size
    height, width = images[0].shape
    num_bands = width // bin_size_x

    full_width = len(images)
    full_cube = np.zeros((height, full_width, num_bands), dtype=np.uint8)

    # Build cube
    for idx, cropped_img in enumerate(images):
        for row in range(height):
            binned_spectrum = spectral_binning(cropped_img[row, :], bin_size=bin_size_x)
            full_cube[row, idx, :] = binned_spectrum

    return full_cube

def normalize_channel(channel):
    """Normalize a single band to 0-255 range for better visualization."""
    norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def extract_rgb_from_cube(cube):
    red_band, green_band, blue_band = load_calibrated_rgb_bands()

    red = cube[:, :, red_band]
    green = cube[:, :, green_band]
    blue = cube[:, :, blue_band]

    # Normalize each channel separately (per-channel stretch)
    red = normalize_channel(red)
    green = normalize_channel(green)
    blue = normalize_channel(blue)

    # Stack into final RGB image
    rgb = np.stack([blue, green, red], axis=2)
    rgb_corrected = correct_perspective(rgb, scale_y=0.6)
    return rgb_corrected.astype(np.uint8)


def auto_select_rgb_bands(cube, start_wavelength=400, end_wavelength=800):
    height, width, num_bands = cube.shape
    wavelengths = np.linspace(start_wavelength, end_wavelength, num_bands)

    avg_spectrum = cube.mean(axis=(0, 1))  # Average over all pixels

    # Band ranges
    blue_range = (wavelengths >= 420) & (wavelengths <= 500)
    green_range = (wavelengths >= 510) & (wavelengths <= 580)
    red_range = (wavelengths >= 600) & (wavelengths <= 700)

    blue_band = np.argmax(avg_spectrum[blue_range])
    green_band = np.argmax(avg_spectrum[green_range])
    red_band = np.argmax(avg_spectrum[red_range])

    # Convert local index back to global band index
    blue_band_idx = np.where(blue_range)[0][blue_band]
    green_band_idx = np.where(green_range)[0][green_band]
    red_band_idx = np.where(red_range)[0][red_band]

    print(f"Auto-selected bands: R:{red_band_idx}, G:{green_band_idx}, B:{blue_band_idx}")
    return red_band_idx, green_band_idx, blue_band_idx



if __name__ == "__main__":
    scan_folder = os.path.join(os.getcwd(), "data", "scan_30April_17:40:21")  # Adjust if needed

    print("Building hyperspectral cube...")
    cube = create_hyperspectral_cube(scan_folder)

    if cube is not None:
        print(f"Cube shape: {cube.shape} (height, width, bands)")

        # Save cube as .npz
        cube_path = os.path.join(scan_folder, "hyperspectral_cube.npz")
        np.savez_compressed(cube_path, cube=cube)
        print(f"Hyperspectral cube saved to {cube_path}")

        # Extract RGB image
        print("Extracting RGB image from cube...")
        rgb = extract_rgb_from_cube(cube)

        # Save and show result
        rgb_path = os.path.join(scan_folder, "rgb_composite.png")
        cv2.imwrite(rgb_path, rgb)
        print(f"RGB composite saved to {rgb_path}")

        cv2.imshow("RGB Composite", rgb)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("❌ Cube build failed.")



#if __name__ == "__main__":
    # Select scan folder
    scan_folder = os.path.join(os.getcwd(), "data", "scan_30April_17:40:21")

    # Step 1: Load one image
    image_files = sorted([f for f in os.listdir(scan_folder) if f.startswith("X") and f.endswith(".png")])
    if not image_files:
        print("No images found.")
        exit()

    first_img_path = os.path.join(scan_folder, 'X1000_Z729.png')
    print(f"Opening image: {first_img_path}")
    img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Crop image
    cropped = crop_useful_area(img)
    height, width = cropped.shape
    print(f"Cropped image shape: {cropped.shape} (Height x Width)")

    # Step 3: Spectral binning row-by-row
    binned_rows = np.array([spectral_binning(cropped[y, :], bin_size=bin_size_x) for y in range(height)])
    print(f"Binned data shape: {binned_rows.shape} (Height x Bands)")

    # Step 4: Print spectrum of a specific row
    row_index = 100
    if row_index < height:
        print(f"Spectrum at row {row_index}:")
        print(binned_rows[row_index])
    else:
        print(f"Row {row_index} is out of bounds for image height {height}")

    # Step 5: Plot the spectrum (optional)
    wavelengths = np.linspace(start_wavelength, end_wavelength, binned_rows.shape[1])
    import matplotlib.pyplot as plt
    plt.plot(wavelengths, binned_rows[row_index])
    plt.title(f"Spectrum at Row {row_index}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.show()
