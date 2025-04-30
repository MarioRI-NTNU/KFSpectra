import sys
import os
import warnings
import time
warnings.filterwarnings("ignore", category=SyntaxWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from printer.printer_control import PrinterController
from camera.capture import Camera

def main():
    printer = PrinterController()
    camera = Camera()

    try:
        printer.connect()
        if not printer.serial:
            print("Printer connection failed. Exiting.")
            return

        camera.connect()

        # Home printer axes
        printer.home()

        # Scan parameters
        start_x = 0
        end_x = 60  # mm
        start_z = 0
        end_z = 38*2  # mm
        step_size_x = 1 # mm per step (both X and Z)
        step_size_z = 38 # mm per step (both X and Z)

        # Create a unique folder for this scan
        scan_time = time.strftime("%d%B_%H:%M:%S")  # Example: 27April_20:27:05
        scan_folder = os.path.join(os.getcwd(), "data", f"scan_{scan_time}")
        os.makedirs(scan_folder, exist_ok=True)
        print(f"Saving scan data to: {scan_folder}")

        # Move to initial position
        printer.move_to(x=start_x, z=start_z)
        time.sleep(1)

        print("Starting full 2D scan...")

        direction = 1  # Start moving in positive X direction

        for z in np.arange(start_z, end_z + 0.001, step_size_z):
            printer.move_to(z=z)
            time.sleep(0.5)

            if direction == 1:
                x_positions = np.arange(start_x, end_x + 0.001, step_size_x)
            else:
                x_positions = np.arange(end_x, start_x - 0.001, -step_size_x)

            for x in x_positions:
                print(f"Capturing frame at X={x:.2f} mm, Z={z:.2f} mm")
                printer.move_to(x=x)
                time.sleep(0.5)

                # Save image
                filename = f"X{int(x*10):03}_Z{int(z*10):03}.png"
                full_path = os.path.join(scan_folder, filename)
                camera.save_frame(full_path)

            direction *= -1  # Reverse X direction for next Z row

        print("Full 2D scan completed successfully.")


    except Exception as e:
        print(f"Error during scanning: {e}")

    finally:
        camera.disconnect()
        printer.disconnect()

if __name__ == "__main__":
    main()
