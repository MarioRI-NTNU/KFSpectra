import sys
import os

from pyueye import ueye
import numpy as np
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class Camera:
    def __init__(self, camera_cfg):
        self.hCam = ueye.HIDS(0)
        self.update_config(camera_cfg)

        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.initialized = False

    def update_config(self, camera_cfg):
        """Update all camera parameters with a new config dictionary."""
        self.exposure_time = camera_cfg["EXPOSURE_TIME_MS"]
        self.gain = camera_cfg["MASTER_GAIN"]
        self.width = camera_cfg["CAMERA_WIDTH"]
        self.height = camera_cfg["CAMERA_HEIGHT"]
        self.bits_per_pixel = camera_cfg["BITS_PER_PIXEL"]
        self.black_level = camera_cfg["BLACK_LEVEL"]
        self.data_dir = camera_cfg["DATA_DIR"]

    def connect(self):
        ret = ueye.is_InitCamera(self.hCam, None)
        if ret != ueye.IS_SUCCESS:
            raise Exception("Failed to initialize camera.")

        print(f"Camera connected with exposure={self.exposure_time}ms, gain={self.gain}, black_level={self.black_level}")

        # Set pixel format: Mono8
        ueye.is_SetColorMode(self.hCam, ueye.IS_CM_MONO8)

        # Set exposure time
        exposure_param = ueye.c_double(self.exposure_time)
        ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_param, 8)

        # Set gain
        ueye.is_SetHardwareGain(self.hCam, self.gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)

        # Set black level
        ueye.is_Blacklevel(self.hCam, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, ueye.c_int(self.black_level), 4)

        # Disable auto features
        ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.DOUBLE(0), ueye.DOUBLE(0))
        ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.DOUBLE(0), ueye.DOUBLE(0))

        # Allocate memory for image
        ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.bits_per_pixel, self.pcImageMemory, self.MemID)
        ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)

        # Start video capture
        ueye.is_CaptureVideo(self.hCam, ueye.IS_WAIT)

        self.initialized = True
        print("Video capture started — camera ready.")

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

    def save_frame(self, filename="debug_picture.png"):
        frame = self.capture_frame()
        os.makedirs(self.data_dir, exist_ok=True)
        full_path = os.path.join(self.data_dir, filename)
        cv2.imwrite(full_path, frame)
        print(f"Image saved at {full_path}")

    def disconnect(self):
        if self.initialized:
            ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
            ueye.is_ExitCamera(self.hCam)
            self.initialized = False
            print("Camera disconnected and resources released.")

if __name__ == "__main__":
    import yaml

    # Example manual load for testing standalone:
    CONFIG_PATH = os.path.join(BASE_DIR, 'edge', 'config.yaml')
    with open(CONFIG_PATH, 'r') as f:
        full_config = yaml.safe_load(f)
    camera_cfg = full_config["camera"]

    cam = Camera(camera_cfg)
    try:
        cam.connect()
        print("Press 'q' to exit, 's' to save an image.")

        save_counter = 1
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
