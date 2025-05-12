import os
import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
from utils import config

def crop_useful_area(img):
    y0, y1 = config.CROP_Y_START, config.CROP_Y_END
    x0, x1 = config.CROP_X_START, config.CROP_X_END
    return img[y0:y1, x0:x1]

def spectral_binning(image_row, bin_size=config.BIN_SIZE_X):
    return np.array([
        np.mean(image_row[i:i+bin_size]) for i in range(0, len(image_row), bin_size)
    ])

def load_calibrated_rgb_bands(calibration_path=config.CALIBRATION_PATH):
    with open(calibration_path, "r") as f:
        calib = json.load(f)
    return calib["red_band"], calib["green_band"], calib["blue_band"]

def correct_perspective(image, scale_y=config.PERSPECTIVE_SCALE_Y):
    height, width = image.shape[:2]
    new_height = int(height * scale_y)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LINEAR)

def create_hyperspectral_cube(scan_folder):
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

    height, width = images[0].shape
    num_bands = width // config.BIN_SIZE_X
    full_width = len(images)
    full_cube = np.zeros((height, full_width, num_bands), dtype=np.uint8)

    for idx, cropped_img in enumerate(images):
        for row in range(height):
            binned = spectral_binning(cropped_img[row, :])
            full_cube[row, idx, :] = binned

    return full_cube

def normalize_channel(channel):
    norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

def extract_rgb_from_cube(cube):
    r, g, b = load_calibrated_rgb_bands()
    red = normalize_channel(cube[:, :, r])
    green = normalize_channel(cube[:, :, g])
    blue = normalize_channel(cube[:, :, b])
    rgb = np.stack([blue, green, red], axis=2)
    return correct_perspective(rgb)

def auto_select_rgb_bands(cube):
    h, w, nb = cube.shape
    wl = np.linspace(config.START_WAVELENGTH, config.END_WAVELENGTH, nb)
    avg = cube.mean(axis=(0, 1))

    r = np.where((wl >= 600) & (wl <= 700))[0][np.argmax(avg[(wl >= 600) & (wl <= 700)])]
    g = np.where((wl >= 510) & (wl <= 580))[0][np.argmax(avg[(wl >= 510) & (wl <= 580)])]
    b = np.where((wl >= 420) & (wl <= 500))[0][np.argmax(avg[(wl >= 420) & (wl <= 500)])]

    print(f"Auto-selected bands: R:{r}, G:{g}, B:{b}")
    return r, g, b

if __name__ == "__main__":
    scan_folder = os.path.join(os.getcwd(), "data", "scan_30April_17:40:21")  # Update if needed

    print("Building hyperspectral cube...")
    cube = create_hyperspectral_cube(scan_folder)

    if cube is not None:
        print(f"Cube shape: {cube.shape} (height, width, bands)")

        save_path = os.path.join(scan_folder, "hyperspectral_cube.npz")
        np.savez_compressed(save_path, cube=cube)
        print(f"Cube saved to {save_path}")

        print("Extracting RGB composite...")
        rgb = extract_rgb_from_cube(cube)

        rgb_path = os.path.join(scan_folder, "rgb_composite.png")
        cv2.imwrite(rgb_path, rgb)
        print(f"RGB image saved to {rgb_path}")

        cv2.imshow("RGB Composite", rgb)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Cube build failed.")



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
