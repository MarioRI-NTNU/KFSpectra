# utils/config.py

import os
import json

# -------------------------
# General Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration.json")

# -------------------------
# Camera Settings
# -------------------------
CAMERA_WIDTH = 1936
CAMERA_HEIGHT = 1216
EXPOSURE_TIME_MS = 92.72
MASTER_GAIN = 24
BLACK_LEVEL = 4
BITS_PER_PIXEL = 8
# -------------------------
# Printer Settings
# -------------------------
PRINTER_PORT = "/dev/ttyACM0"
PRINTER_BAUDRATE = 115200
PRINTER_FEEDRATE = 1500
SERIAL_TIMEOUT = 2

# -------------------------
# Hyperspectral Settings
# -------------------------
BIN_SIZE_X = 8
START_WAVELENGTH = 400  # nm
END_WAVELENGTH = 800    # nm
CROP_Y_START = 296
CROP_Y_END = 845
CROP_X_START = 0
CROP_X_END = CAMERA_WIDTH

# -------------------------
# Scanning Settings
# -------------------------
SCAN_START_X = 100
SCAN_END_X = 100
SCAN_START_Z = 20
SCAN_END_Z = 80
STEP_SIZE_X = 10
STEP_SIZE_Z = 0.2
PAUSE_AFTER_MOVE = 0.5  # seconds

# -------------------------
# Processing Settings
# -------------------------
def load_rgb_bands():
    try:
        with open(CALIBRATION_FILE, "r") as f:
            calib = json.load(f)
        return calib["red_band"], calib["green_band"], calib["blue_band"]
    except (FileNotFoundError, KeyError):
        print("Calibration file not found or invalid. Using defaults.")
        return None, None, None

# Binning and wavelength
BIN_SIZE_X = 8
START_WAVELENGTH = 400
END_WAVELENGTH = 800

# Crop coordinates
CROP_Y_START = 296
CROP_Y_END = 845
CROP_X_START = 0
CROP_X_END = 1936

# Perspective correction
PERSPECTIVE_SCALE_Y = 0.6

# Calibration file path
CALIBRATION_PATH = "calibration.json"