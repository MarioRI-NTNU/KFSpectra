import os
import cv2
import numpy as np

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
    height, width, num_bands = cube.shape

    wavelengths = np.linspace(start_wavelength, end_wavelength, num_bands)

    red_band = np.argmin(np.abs(wavelengths - 700))
    green_band = np.argmin(np.abs(wavelengths - 550))
    blue_band = np.argmin(np.abs(wavelengths - 450))

    print(f"Using bands -> R:{red_band}, G:{green_band}, B:{blue_band}")

    # Extract bands
    red = cube[:, :, red_band]
    green = cube[:, :, green_band]
    blue = cube[:, :, blue_band]

    # --- NEW: Normalize each channel ---
    red_norm = normalize_channel(red)
    green_norm = normalize_channel(green)
    blue_norm = normalize_channel(blue)

    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
    return rgb_image


if __name__ == "__main__":
    # Select folder
    scan_folder = os.path.join(os.getcwd(), "data", "scan_28April_15:06:11")

    # Create hyperspectral cube
    print("Building hyperspectral cube...")
    cube = create_hyperspectral_cube(scan_folder)

    if cube is not None:
        print(f"Cube shape: {cube.shape}")  # (Height, Width, Bands)

        # Save cube
        save_path = os.path.join(scan_folder, "hyperspectral_cube.npz")
        np.savez_compressed(save_path, cube=cube)
        print(f"Hyperspectral cube saved at {save_path}")

        # Extract RGB
        print("Extracting RGB composite image...")
        rgb_image = extract_rgb_from_cube(cube)

        # Save RGB image
        rgb_save_path = os.path.join(scan_folder, "rgb_composite.png")
        cv2.imwrite(rgb_save_path, rgb_image)
        print(f"RGB composite saved at {rgb_save_path}")

        # Show RGB image
        cv2.imshow("RGB Composite", rgb_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Cube was not built.")
