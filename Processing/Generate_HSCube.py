import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
from utils import config
from spectral import envi


def crop_useful_area(img):
    y0, y1 = config.CROP_Y_START, config.CROP_Y_END
    x0, x1 = config.CROP_X_START, config.CROP_X_END
    return img[y0:y1, x0:x1]

def spectral_binning(image, bin_size=8):
    width = image.shape[1]
    trimmed_width = (width // bin_size) * bin_size
    image = image[:, :trimmed_width]  # remove trailing pixels if needed
    binned = image.reshape(image.shape[0], -1, bin_size).mean(axis=2)
    return binned

def load_calibrated_rgb_bands(calibration_path=config.CALIBRATION_PATH):
    with open(calibration_path, "r") as f:
        calib = json.load(f)
    return calib["red_band"], calib["green_band"], calib["blue_band"]

def correct_perspective(image, scale_y=config.PERSPECTIVE_SCALE_Y):
    height, width = image.shape[:2]
    new_height = int(height * scale_y)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LINEAR)

def create_hyperspectral_cube(scan_folder):
    cropped_images = []

    for filename in sorted(os.listdir(scan_folder)):
        if filename.endswith(".png") and filename.startswith("X"):
            path = os.path.join(scan_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                cropped = crop_useful_area(img)
                binned = spectral_binning(cropped)  # bin the whole image
                cropped_images.append(binned)

    if not cropped_images:
        print("No images found!")
        return None

    # Stack images along the X axis
    cube = np.stack(cropped_images, axis=1)  # (rows, images, bands)
    return cube.astype(np.uint8)


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

def export_cube_envi(cube, output_path, wavelengths=None):
    height, width, bands = cube.shape
    cube_envi = cube.transpose(2, 0, 1)  # (bands, rows, columns)

    metadata = {
        'lines': height,
        'samples': width,
        'bands': bands,
        'interleave': 'bsq',
        'data type': 1,  # 1 = uint8
        'byte order': 0,
        'sensor type': 'KFSpectra_HSI',
    }

    if wavelengths is not None:
        metadata['wavelength'] = list(map(str, wavelengths))
        metadata['wavelength units'] = 'nm'

    envi.save_image(output_path + '.hdr', cube_envi, metadata=metadata, dtype=np.uint8, force=True)


if __name__ == "__main__":
    scan_folder = os.path.join(os.getcwd(), "data", "scan_30April_17:40:21")  # Update as needed

    print("Building hyperspectral cube...")
    cube = create_hyperspectral_cube(scan_folder)

    if cube is not None:
        print(f"Cube shape: {cube.shape} (height, width, bands)")

        # Save compressed NumPy cube (optional)
        if config.SAVE_NPZ_CUBE:
            npz_path = os.path.join(scan_folder, "hyperspectral_cube.npz")
            np.savez_compressed(npz_path, cube=cube)
            print(f"Cube saved to {npz_path}")

        # Save ENVI .hdr + .dat or .img
        if config.SAVE_ENVI_CUBE:
            from spectral import envi
            def export_cube_envi(cube, output_path, wavelengths=None):
                h, w, b = cube.shape
                cube_envi = cube.transpose(2, 0, 1)
                metadata = {
                    'lines': h,
                    'samples': w,
                    'bands': b,
                    'interleave': 'bsq',
                    'data type': 1,
                    'byte order': 0,
                    'sensor type': 'KFSpectra_HSI',
                }
                if wavelengths is not None:
                    metadata['wavelength'] = list(map(str, wavelengths))
                    metadata['wavelength units'] = 'nm'
                envi.save_image(output_path + '.hdr', cube_envi, metadata=metadata, dtype=np.uint8, force=True)

                if config.ENVI_EXTENSION != ".img":
                    os.rename(output_path + ".img", output_path + config.ENVI_EXTENSION)

            envi_base = os.path.join(scan_folder, "hyperspectral_cube_envi")
            wavelengths = np.linspace(config.START_WAVELENGTH, config.END_WAVELENGTH, cube.shape[2])
            export_cube_envi(cube, envi_base, wavelengths=wavelengths)
            print(f"ENVI cube saved to {envi_base}{config.ENVI_EXTENSION} and .hdr")

        # Extract and save RGB
        if config.SAVE_RGB_PNG:
            print("Extracting RGB composite...")
            rgb = extract_rgb_from_cube(cube)
            rgb_path = os.path.join(scan_folder, "rgb_composite.png")
            cv2.imwrite(rgb_path, rgb)
            print(f"RGB image saved to {rgb_path}")
            cv2.imshow("RGB Composite", rgb)
            print("Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
