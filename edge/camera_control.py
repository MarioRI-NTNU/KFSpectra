import sys
import os
import numpy as np
import cv2
from ids_peak import ids_peak
import ctypes


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class Camera:
    def __init__(self, camera_cfg):
        self.update_config(camera_cfg)
        ids_peak.Library.Initialize()
        self.device_manager = ids_peak.DeviceManager.Instance()
        self.device = None
        self.datastream = None
        self.nodemap = None
        self.initialized = False

    def update_config(self, camera_cfg):
        self.exposure_time = camera_cfg["EXPOSURE_TIME_MS"] * 1000  # µs
        self.gain = camera_cfg["MASTER_GAIN"]
        self.width = camera_cfg["CAMERA_WIDTH"]
        self.height = camera_cfg["CAMERA_HEIGHT"]
        self.data_dir = camera_cfg["DATA_DIR"]
        self.binning_factor = camera_cfg.get("BINNING_FACTOR", 2)
        self.roi_top = 248
        self.roi_bottom = 803

    def connect(self):
        self.device_manager.Update()
        if self.device_manager.Devices().empty():
            raise Exception("No camera found!")

        print(f"[INFO] Found device: {self.device_manager.Devices()[0].ModelName()}")

        self.device = self.device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self.datastream = self.device.DataStreams()[0].OpenDataStream()
        self.nodemap = self.device.RemoteDevice().NodeMaps()[0]

        # Configure
        self.nodemap.FindNode("ExposureTime").SetValue(self.exposure_time)
        self.nodemap.FindNode("Gain").SetValue(self.gain)

        width = int(self.nodemap.FindNode("Width").Value())
        height = int(self.nodemap.FindNode("Height").Value())
        bpp = 1  # Mono8 pixel
        payload_size = width * height * bpp

        print(f"[INFO] Config: Exposure={self.exposure_time} µs Gain={self.gain} → Frame {width}x{height}")

        # Announce & queue buffers
        self.buffers = []
        for _ in range(5):
            buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
            self.buffers.append(buffer)
            self.datastream.QueueBuffer(buffer)

       # Start acquisition
        self.datastream.StartAcquisition()
        self.nodemap.FindNode("AcquisitionStart").Execute()


        self.initialized = True
        print("[INFO] Acquisition started — ready to capture.\n")

    def capture_frame(self):
        if not self.initialized:
            raise Exception("Camera not initialized!")

        buffer = self.datastream.WaitForFinishedBuffer(5000)
        width = buffer.Width()
        height = buffer.Height()

        import ctypes
        buf_ptr = int(buffer.BasePtr())  # Correctly convert to int!
        ptr = ctypes.cast(buf_ptr, ctypes.POINTER(ctypes.c_ubyte))
        img_array = np.ctypeslib.as_array(ptr, shape=(height, width))
        img_copy = np.copy(img_array)

        self.datastream.QueueBuffer(buffer)

        print(f"[INFO] Frame shape: {img_copy.shape} dtype: {img_copy.dtype} mean: {np.mean(img_copy):.2f}")
        return img_copy



    def bin_image(self, image):
        factor = self.binning_factor  # 8
        h, w = image.shape
        new_w = w // factor
        image = image[:, :new_w * factor]  # Trim width to multiple of factor
        binned = image.reshape(h, new_w, factor).mean(axis=2)
        return binned.astype(np.uint8)


    def crop_roi(self, image):
        return image[self.roi_top:self.roi_bottom, :]

    def save_frame(self, file_name="debug_picture.png"):
        frame = self.capture_frame()
        cropped = self.crop_roi(frame)
        binned = self.bin_image(cropped)

        out_dir = os.path.abspath(os.path.join(BASE_DIR, self.data_dir))
        os.makedirs(out_dir, exist_ok=True)

        output_path = os.path.join(out_dir, f"{file_name}")
        cv2.imwrite(output_path, binned)
        print(f"[INFO] Final saved frame shape: {cropped.shape} → binned shape: {binned.shape}")


        os.sync()


    def disconnect(self):
        if self.initialized:
            self.nodemap.FindNode("AcquisitionStop").Execute()
            self.datastream.StopAcquisition()
            ids_peak.Library.Close()
            self.initialized = False
            print("[INFO] Camera disconnected cleanly.")

if __name__ == "__main__":
    import yaml

    CONFIG_PATH = os.path.join(BASE_DIR, 'edge', 'config.yaml')
    with open(CONFIG_PATH, 'r') as f:
        full_config = yaml.safe_load(f)
    camera_cfg = full_config["camera"]

    cam = Camera(camera_cfg)

    try:
        cam.connect()
        cam.save_frame()
    finally:
        cam.disconnect()
