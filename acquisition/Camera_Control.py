import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pyueye import ueye
import numpy as np
import cv2
import time
from utils import config

class Camera:
    def __init__(self):
        self.hCam = ueye.HIDS(0)
        self.exposure_time = config.EXPOSURE_TIME_MS
        self.gain = config.MASTER_GAIN
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT
        self.bits_per_pixel = config.BITS_PER_PIXEL # MONO8 = 8 bits
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.initialized = False

    def connect(self):
        ret = ueye.is_InitCamera(self.hCam, None)
        if ret != ueye.IS_SUCCESS:
            raise Exception("Failed to initialize camera.")

        # Set pixel format: Mono8
        ueye.is_SetColorMode(self.hCam, ueye.IS_CM_MONO8)

        # Exposure time
        exposure_param = ueye.c_double(self.exposure_time)
        ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_param, 8)

        # Gain
        ueye.is_SetHardwareGain(self.hCam, self.gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)

        # Black level
        ueye.is_Blacklevel(self.hCam, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, ueye.c_int(config.BLACK_LEVEL), 4)

        # Disable auto features
        ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.DOUBLE(0), ueye.DOUBLE(0))
        ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.DOUBLE(0), ueye.DOUBLE(0))

        # Allocate memory for image
        ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.bits_per_pixel, self.pcImageMemory, self.MemID)
        ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)

        # Start video capture
        ueye.is_CaptureVideo(self.hCam, ueye.IS_WAIT)

        self.initialized = True
        print("Camera initialized and video capture started.")

    def capture_frame(self):
        if not self.initialized:
            raise Exception("Camera not initialized.")

        imageData = ueye.get_data(
            self.pcImageMemory,
            self.width,
            self.height,
            self.bits_per_pixel,
            self.width,
            True
        )
        image = np.reshape(imageData, (self.height, self.width))

        # Normalize contrast
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Flip horizontally
        image = cv2.flip(image, 1)

        return image

    def save_frame(self, filename="captured_frame.png"):
        frame = self.capture_frame()
        os.makedirs(config.DATA_DIR, exist_ok=True)
        full_path = os.path.join(config.DATA_DIR, filename)
        cv2.imwrite(full_path, frame)
        print(f"Image saved at {full_path}")

    def disconnect(self):
        if self.initialized:
            ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
            ueye.is_ExitCamera(self.hCam)
            self.initialized = False
            print("Camera disconnected and resources released.")

if __name__ == "__main__":#Display camera feed and save pictures when S key is pressed
    cam = Camera()
    try:
        cam.connect()
        print("Press 'q' to exit, 's' to save an image.")

        save_counter = 1  # Start counting from 1

        while True:
            frame = cam.capture_frame()
            cv2.imshow("Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"captured_frame_{save_counter}.png"
                cam.save_frame(filename)
                save_counter += 1

        cv2.destroyAllWindows()
    finally:
        cam.disconnect()
