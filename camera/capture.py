from pyueye import ueye
import numpy as np
import cv2
import time
import os

class Camera:
    def __init__(self, exposure_time=100.0, gain=20, width=1936, height=1216):
        self.hCam = ueye.HIDS(0)
        self.exposure_time = exposure_time
        self.gain = gain
        self.width = width
        self.height = height
        self.bits_per_pixel = 8  # MONO8 = 8 bits
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.initialized = False

    def connect(self):
        ret = ueye.is_InitCamera(self.hCam, None)
        if ret != ueye.IS_SUCCESS:
            raise Exception("Failed to initialize camera.")

        # Set color mode to MONO8 (grayscale) for hyperspectral
        ueye.is_SetColorMode(self.hCam, ueye.IS_CM_MONO8)

        # Set exposure
        exposure_param = ueye.c_double(self.exposure_time)
        ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_param, 8)

        # Set hardware gain
        ueye.is_SetHardwareGain(self.hCam, self.gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)

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

        imageData = ueye.get_data(self.pcImageMemory, self.width, self.height, self.bits_per_pixel, self.width, True)
        image = np.reshape(imageData, (self.height, self.width))

        # Optional: Apply contrast normalization
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        return image

    def save_frame(self, filename="captured_frame.png"):
        frame = self.capture_frame()
        
        # Ensure the data directory exists
        save_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(save_dir, exist_ok=True)

        # Full path
        full_path = os.path.join(save_dir, filename)

        cv2.imwrite(full_path, frame)
        print(f"Image saved at {full_path}")

    def disconnect(self):
        if self.initialized:
            ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
            ueye.is_ExitCamera(self.hCam)
            self.initialized = False
            print("Camera disconnected and resources released.")

if __name__ == "__main__":
    cam = Camera()
    try:
        cam.connect()
        print("Press 'q' to exit, 's' to save an image.")
        while True:
            frame = cam.capture_frame()
            cv2.imshow("Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cam.save_frame()

        cv2.destroyAllWindows()

    finally:
        cam.disconnect()
