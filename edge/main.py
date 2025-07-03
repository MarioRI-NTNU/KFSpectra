import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from mqtt import HSI_MQTT

def main():
    hsi_mqtt = HSI_MQTT()
    hsi_mqtt.connect()
    hsi_mqtt.loop_forever()

if __name__ == "__main__":
    main()
