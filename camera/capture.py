from pyueye import ueye
import numpy as np
import cv2

# Initialize the camera
hCam = ueye.HIDS(0)  # First available camera
ret = ueye.is_InitCamera(hCam, None)

if ret != ueye.IS_SUCCESS:
    print("❌ Error: Could not initialize camera")
    exit()

# Set color mode (adjust based on Peak Cockpit settings)
ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8)  # IS_CM_BGR8_PACKED for color

# Set Exposure (increase if image is dark)
desired_exposure_time = 100.0  # Adjust as needed
ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.c_double(desired_exposure_time), 8)

# Set Gain (increase if image is dark)
desired_gain = 20  # Adjust as needed
ueye.is_SetHardwareGain(hCam, desired_gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)

# Allocate memory
sensor_width = 1936  # Camera resolution
sensor_height = 1216
pcImageMemory = ueye.c_mem_p()
MemID = ueye.int()
ueye.is_AllocImageMem(hCam, sensor_width, sensor_height, 8, pcImageMemory, MemID)
ueye.is_SetImageMem(hCam, pcImageMemory, MemID)

# Start Capturing
ueye.is_CaptureVideo(hCam, ueye.IS_WAIT)

print("📷 Camera initialized. Press 'q' to exit, 's' to save an image.")

# Real-time Display Loop
while True:
    imageData = ueye.get_data(pcImageMemory, sensor_width, sensor_height, 8, sensor_width, True)
    image = np.reshape(imageData, (sensor_height, sensor_width))

    # Apply contrast enhancement (optional)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Live Feed", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q'
        break
    elif key == ord('s'):  # Save image on 's'
        cv2.imwrite("captured_frame.png", image)
        print("✅ Image saved as 'captured_frame.png'")

# Cleanup
ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)
ueye.is_ExitCamera(hCam)
cv2.destroyAllWindows()

print("🛑 Camera closed.")
