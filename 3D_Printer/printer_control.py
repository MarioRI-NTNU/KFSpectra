import serial
import time

# Connect to the printer's serial port
printer = serial.Serial('/dev/ttyACM0', 115200, timeout=2)  # Adjust port as needed
time.sleep(2)  # Give time to initialize

# Function to send G-code command
def send_gcode(cmd):
    printer.write((cmd + '\n').encode())
    response = printer.readline().decode().strip()
    print(f"Sent: {cmd} | Response: {response}")
    return response

# Wake up the printer
send_gcode("M17")  # Enable motors

# Move to a known position (e.g., home then move X and Z)
send_gcode("G28")  # Home all axes

# Close serial connection
printer.close()