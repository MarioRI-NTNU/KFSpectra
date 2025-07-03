import sys
import os
import serial
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class Printer:
    def __init__(self, printer_cfg):
        """Initialize printer with config dictionary."""
        self.update_config(printer_cfg)
        self.serial = None

    def update_config(self, printer_cfg):
        """Update printer config fields on the fly."""
        self.device = printer_cfg["DEVICE"]
        self.baudrate = printer_cfg.get("BAUDRATE", 115200)
        self.timeout = printer_cfg.get("TIMEOUT", 2)
        self.steps_per_mm = printer_cfg.get("STEPS_PER_MM", 80)
        self.extruder_temp = printer_cfg.get("EXTRUDER_TEMP", 200)
        self.default_feedrate = printer_cfg.get("DEFAULT_FEEDRATE", 1200)  # Added

    def connect(self):
        try:
            self.serial = serial.Serial(self.device, self.baudrate, timeout=self.timeout)
            time.sleep(2)

            if self.serial.is_open:
                print(f"Printer connected on {self.device} at {self.baudrate} baud.")
            else:
                raise Exception("Printer port opened but not active.")

            print("Requesting firmware info...")
            self.serial.write(b'M115\n')
            time.sleep(0.1)

            firmware_received = False
            start_time = time.time()
            while time.time() - start_time < 5:
                response = self.serial.readline().decode(errors='ignore').strip()
                if response:
                    print(f"Response: {response}")
                    if "firmware" in response.lower() or "machine" in response.lower() or "prusa" in response.lower():
                        firmware_received = True
                        break
                    if "crash detected" in response.lower() or "error" in response.lower() or "cold" in response.lower():
                        raise Exception("Printer reported error state. Check if PSU is ON.")

            if not firmware_received:
                print("Warning: Printer responded but firmware info was not confirmed. Proceeding cautiously.")

            print("Enabling motors...")
            self.serial.write(b'M17\n')
            time.sleep(0.5)

            print("Testing small X move...")
            self.serial.write(b'G91\n')
            time.sleep(0.1)
            self.serial.write(f'G1 X1 F{self.default_feedrate}\n'.encode())
            time.sleep(0.1)
            self.serial.write(b'G90\n')

            start_time = time.time()
            move_successful = False
            while time.time() - start_time < 10:
                response = self.serial.readline().decode(errors='ignore').strip()
                if response:
                    print(f"Response: {response}")
                    if "ok" in response.lower():
                        move_successful = True
                        break
                    if "crash detected" in response.lower() or "error" in response.lower() or "cold" in response.lower():
                        raise Exception("Movement failed or printer crashed during connection check.")

            if not move_successful:
                raise Exception("Printer did not confirm movement readiness.")

            print("Printer ready for scanning operations.")

        except serial.SerialException as e:
            print(f"Failed to connect to the printer: {e}")
            self.serial = None
        except Exception as e:
            print(f"An unexpected error occurred during connection: {e}")
            self.serial = None

    def send_gcode(self, cmd, wait=False):
        if self.serial:
            try:
                print(f"\nSending: {cmd}")
                self.serial.write((cmd + '\n').encode())
                time.sleep(0.1)

                start_time = time.time()
                while time.time() - start_time < 10:
                    response = self.serial.readline().decode(errors='ignore').strip()
                    if response:
                        if not response.startswith("echo:busy"):
                            print(f"Response: {response}")
                        if "ok" in response.lower():
                            break

                if wait:
                    self.serial.write(b'M400\n')
                    time.sleep(0.1)
                    while True:
                        response = self.serial.readline().decode(errors='ignore').strip()
                        if response:
                            if not response.startswith("echo:busy"):
                                print(f"Wait M400 response: {response}")
                            if "ok" in response.lower():
                                break

            except serial.SerialException as e:
                print(f"Serial error: {e}")
        else:
            raise Exception("Serial connection not established")

    def wait_until_ready(self, timeout=10):
        if not self.serial:
            raise Exception("Serial connection not established")

        print("Waiting for printer to finish...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.serial.readline().decode(errors='ignore').strip()
            if response:
                if not response.startswith("echo:busy"):
                    print(f"Wait response: {response}")
                if "ok" in response.lower():
                    print("Printer is ready.")
                    return
            time.sleep(0.1)

        raise Exception("Timeout waiting for printer to become ready.")

    def move_to(self, x=None, z=None, feedrate=None):
        if feedrate is None:
            feedrate = self.default_feedrate

        cmd = "G1"
        if x is not None:
            cmd += f" X{x}"
        if z is not None:
            cmd += f" Z{z}"
        cmd += f" F{feedrate}"
        self.send_gcode(cmd, wait=True)

    def home(self):
        self.send_gcode("G1 Z15", wait=True)
        print("Homing X...")
        self.send_gcode("G28 X0", wait=True)
        print("Homing Z...")
        self.send_gcode("G28 Z0", wait=True)
        self.send_gcode("G90", wait=False)
        print("Printer homed!")

    def disconnect(self):
        if self.serial:
            self.serial.close()
            self.serial = None
            print("Printer disconnected")

if __name__ == "__main__":
    import yaml

    CONFIG_PATH = os.path.join(BASE_DIR, 'edge', 'config.yaml')
    with open(CONFIG_PATH, 'r') as f:
        full_config = yaml.safe_load(f)
    printer_cfg = full_config["printer"]

    printer = Printer(printer_cfg)
    printer.connect()

    if printer.serial:
        printer.home()
        while True:
            GcodeX, GcodeZ = input("X:"), input("Z:")
            if GcodeX == "exit":
                break
            printer.move_to(x=GcodeX, z=GcodeZ)

        printer.disconnect()
    else:
        print("Printer connection failed. Exiting program.")
