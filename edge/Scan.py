import sys
import os
import warnings
import time
import numpy as np
warnings.filterwarnings("ignore", category=SyntaxWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from edge.printer_control import Printer
from edge.camera_control import Camera
from utils import config

def main():
    printer = Printer()
    camera = Camera()

    try:
        printer.connect()
        if not printer.serial:
            print("Printer connection failed. Exiting.")
            return

        camera.connect()

        # Home printer
        printer.home()

        # Scan parameters
        start_x = config.SCAN_START_X
        end_x = config.SCAN_END_X
        start_z = config.SCAN_START_Z
        end_z = config.SCAN_END_Z
        step_size_x = config.STEP_SIZE_X
        step_size_z = config.STEP_SIZE_Z

        # Create a scan folder
        scan_time = time.strftime("%d%B_%H:%M:%S")
        scan_folder = os.path.join(os.getcwd(), "data", f"scan_{scan_time}")
        os.makedirs(scan_folder, exist_ok=True)
        print(f"Saving scan data to: {scan_folder}")

        # Move to starting point
        printer.move_to(x=start_x, z=start_z)
        time.sleep(config.PAUSE_AFTER_MOVE)

        print("Starting full 2D scan...")
        direction = 1

        for z in np.arange(start_z, end_z + 0.001, step_size_z):
            printer.move_to(z=z)
            time.sleep(config.PAUSE_AFTER_MOVE)

            x_positions = (
                np.arange(start_x, end_x + 0.001, step_size_x)
                if direction == 1
                else np.arange(end_x, start_x - 0.001, -step_size_x)
            )

            for x in x_positions:
                print(f"Capturing frame at X={x:.2f} mm, Z={z:.2f} mm")
                printer.move_to(x=x)
                time.sleep(config.PAUSE_AFTER_MOVE)

                filename = f"X{int(x*10):03}_Z{int(z*10):03}.png"
                full_path = os.path.join(scan_folder, filename)
                camera.save_frame(full_path)

            direction *= -1

        print("Full 2D scan completed successfully.")

    except Exception as e:
        print(f"Error during scanning: {e}")

    finally:
        camera.disconnect()
        printer.disconnect()

if __name__ == "__main__":
    main()

