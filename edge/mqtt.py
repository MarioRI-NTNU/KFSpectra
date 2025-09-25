import os
import sys
import yaml
import json
import time
import datetime
import glob
import subprocess

import paho.mqtt.client as mqtt
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from camera_control import Camera
from printer_control import Printer

CONFIG_PATH = os.path.join(BASE_DIR, 'edge', 'config.yaml')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

class HSI_MQTT:
    def __init__(self):
        self.config = load_config()
        self.client = mqtt.Client()

        self.broker = self.config["mqtt"]["broker"]
        self.port = self.config["mqtt"]["port"]
        self.topics = self.config["mqtt"]["topics"]

        # Initialize camera and printer with initial config (can be reloaded later)
        self.printer = Printer(self.config["printer"])
        self.cam = Camera(self.config["camera"])
        
        #Avoid resetting for each command. 
        self.printer.connect()
        #Home once - known reference point
        #self.printer.home()

    def connect(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        print(f"Connecting to MQTT broker at {self.broker}:{self.port} ...")
        self.client.connect(self.broker, self.port, 60)

        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to MQTT broker.")
        self.client.subscribe(self.topics["scan_command"])
        self.client.subscribe(self.topics["camera_picture"])
        self.client.subscribe(self.topics["printer_gcode"])
        self.client.subscribe(self.topics["config_request"])

        # Publish current config once on connect
        self.publish_status({"config": self.config})
        self.publish_camera_status()
        self.publish_printer_status()
        print("Published current config on connect.\n")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        print(f"Received message on {topic}: {payload}")

        if topic == self.topics["scan_command"]:
            self.handle_scan_command(payload)
        elif topic == self.topics["camera_picture"]:
            self.handle_camera_picture(payload)
        elif topic == self.topics["printer_gcode"]:
            self.handle_printer_gcode(payload)
        elif topic == self.topics["config_request"]:
            self.handle_config_update(payload)

    def cleanup_old_scans(self, data_dir, keep=5):
        scans = sorted(glob.glob(os.path.join(data_dir, "scan_*")), key=os.path.getmtime)
        if len(scans) > keep:
            for old_scan in scans[:-keep]:
                print(f"Deleting old scan folder: {old_scan}")
                subprocess.run(["rm", "-rf", old_scan])

        tarballs = glob.glob(os.path.join(data_dir, "scan_*.tar.gz"))
        for tb in tarballs:
            os.remove(tb)

    def handle_scan_command(self, payload):
        config = load_config()
        ssh_cfg = config["ssh"]
        camera_cfg = config["camera"]
        printer_cfg = config["printer"]

        self.printer = Printer(printer_cfg)
        self.cam = Camera(camera_cfg)

        print("Starting scan...\n")
        self.publish_status({"status": "scanning"})

        scan_folder = None  

        try:
            self.printer.connect()
            if not self.printer.serial:
                print("Printer connection failed. Exiting.")
                self.publish_status({"status": "error"})
                return

            self.cam.connect()

            # Home printer
            self.printer.home()

            # Scan parameters
            start_x = printer_cfg["X_START"]
            end_x = printer_cfg["X_END"]
            start_z = printer_cfg["Z_START"]
            end_z = printer_cfg["Z_END"]
            step_size_x = printer_cfg["X_STEP"]
            step_size_z = printer_cfg["Z_STEP"]

            # Create scan folder
            scan_time = time.strftime("%d%B_%H:%M:%S")
            scan_folder = os.path.join(os.getcwd(), "data", f"scan_{scan_time}")
            os.makedirs(scan_folder, exist_ok=True)
            print(f"Saving scan data to: {scan_folder}")

            # Move to starting point
            self.printer.move_to(x=start_x, z=start_z)
            time.sleep(0.5)

            print("Starting full 2D scan...")
            direction = 1

            for z in np.arange(start_z, end_z + 0.001, step_size_z):
                self.printer.move_to(z=z)
                time.sleep(0.5)

                x_positions = (
                    np.arange(start_x, end_x + 0.001, step_size_x)
                    if direction == 1
                    else np.arange(end_x, start_x - 0.001, -step_size_x)
                )

                for x in x_positions:
                    print(f"Capturing frame at X={x:.2f} mm, Z={z:.2f} mm")
                    self.printer.move_to(x=x)
                    time.sleep(0.5)

                    filename = f"X{int(x*10):03}_Z{int(z*10):03}.png"
                    full_path = os.path.join(scan_folder, filename)
                    self.cam.save_frame(full_path)

                direction *= -1

            print("Full 2D scan completed successfully.")
            self.publish_status({"status": "idle"})

        except Exception as e:
            print(f"[ERROR] scan failed: {e}\n")
            self.publish_status({"status": "error"})
            return  # prevents accessing undefined scan_folder

        # Only run these if scan was successful
        if scan_folder:
            try:
                tarball = f"{scan_folder}.tar.gz"
                subprocess.run(
                    ["tar", "-czf", tarball, "-C", os.path.dirname(scan_folder), os.path.basename(scan_folder)],
                    check=True
                )
                print(f"Created tarball: {tarball}\n")

                # SCP transfer
                scp_cmd = [
                    "scp",
                    tarball,
                    f"{ssh_cfg['user']}@{ssh_cfg['server_ip']}:{ssh_cfg['dest_folder_scan']}/"
                ]
                print(f"Running SCP command: {' '.join(scp_cmd)}\n")
                subprocess.run(scp_cmd, check=True)
                print(f"Scan tarball sent to {ssh_cfg['server_ip']}:{ssh_cfg['dest_folder_scan']}\n")

                self.publish_status({
                    "status": "idle",
                    "scan_tarball": tarball
                })

            except Exception as e:
                print(f"[ERROR] packaging or transfer failed: {e}\n")
                self.publish_status({"status": "error"})

        
    def handle_camera_picture(self, payload):
        print("Taking debug camera picture...")

        config = load_config()
        camera_cfg = config["camera"]
        ssh_cfg = config["ssh"]

        # Always make DATA_DIR absolute
        data_dir = camera_cfg.get("DATA_DIR", "data")
        data_dir_abs = os.path.abspath(os.path.join(BASE_DIR, data_dir))
        output_name = "debug_picture.png"

        # Actual file the Camera class will write
        local_picture = os.path.join(data_dir_abs, output_name)

        print(f"DATA_DIR from config: {data_dir}")
        print(f"Resolved absolute DATA_DIR: {data_dir_abs}")
        print(f"Full local_picture path: {local_picture}\n")

        try:
            # Try the real camera
            self.cam = Camera(camera_cfg)
            self.cam.connect()
            self.cam.save_frame(file_name=output_name)
            self.cam.disconnect()
            print(f"Camera capture saved at {local_picture}\n")

        except Exception as e:
            print(f"[WARNING] Camera not connected — using fallback image. Reason: {e}\n")
            # If the camera fails, ensure fallback `debug_picture.png` exists

        try:
            # Push to the HA server using SCP
            scp_cmd = [
                "scp",
                local_picture,
                f"{ssh_cfg['user']}@{ssh_cfg['server_ip']}:{ssh_cfg['dest_folder']}/latest_debug_picture.png"
            ]
            print(f"Running SCP command: {' '.join(scp_cmd)}\n")
            subprocess.run(scp_cmd, check=True)
            print(f"Debug picture sent to {ssh_cfg['server_ip']}:{ssh_cfg['dest_folder']}\n")

            self.publish_camera_status({"picture_sent": "true"})

        except Exception as e:
            print(f"[ERROR] SCP push failed: {e}\n")
            self.publish_status({"status": "error"})

    def handle_printer_gcode(self, payload):
        print("Running printer GCode...\n")
        try:
            cmd = payload.get("gcode")
            if not cmd: 
                print("No GCode provided. \n")
                return
            #config = load_config()
            #printer_cfg = config["printer"]
            #self.printer = Printer(printer_cfg)
            #cmd = payload.get("gcode")
            #if not cmd:
            #    print("No GCode provided.\n")
            #    return

            #only do this if we have a closed connection
            if not self.printer.serial or not self.printer.serial.is_open:
                self.printer.connect() 
                
            #self.printer.connect()
            #self.printer.send_gcode(cmd, wait=True)
            #self.printer.disconnect()

            #self.publish_printer_status({"last_gcode": cmd})

            # valgfri "home"-alias
            if cmd.strip().upper() in ("HOME", "G28", "G28 X0 Z0"):
                self.printer.home()
            else:
                self.printer.send_gcode(cmd, wait=True)

            self.publish_printer_status({"last_gcode": cmd})

        except Exception as e:
            print(f"Printer GCode error: {e}\n")
            self.publish_status({"status": "error"})

    def handle_config_update(self, payload):
        print("Updating config.yaml...\n")
        try:
            new_config = payload.get("config")
            if not new_config:
                print("No config provided.\n")
                return

            config = load_config()

            for section, params in new_config.items():
                if section in config:
                    config[section].update(params)
                else:
                    config[section] = params

            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f)

            print("Config updated.\n")
            self.publish_status({"config": "updated"})
            self.publish_status({"config": self.config})
            self.publish_camera_status()
            self.publish_printer_status()

        except Exception as e:
            print(f"Config update error: {e}")
            self.publish_status({"status": "error"})

    def publish_status(self, extra={}):
        data = {"status": extra.get("status", "idle")}
        data.update(extra)
        self.client.publish(self.topics["status"], json.dumps(data))
        print(f"Published scanner status: {data}\n")

    def publish_camera_status(self, extra={}):
        data = {"status": extra.get("status", "idle")}
        data.update(load_config()["camera"])
        data.update(extra)
        self.client.publish(self.topics["camera_status"], json.dumps(data))
        print(f"Published camera status: {data}\n")

    def publish_printer_status(self, extra={}):
        data = {"status": extra.get("status", "idle")}
        data.update(load_config()["printer"])
        data.update(extra)
        self.client.publish(self.topics["printer_status"], json.dumps(data))
        print(f"Published printer status: {data}\n")

    def loop_forever(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping MQTT client...\n")
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    hsi = HSI_MQTT()
    hsi.connect()
    hsi.loop_forever()
