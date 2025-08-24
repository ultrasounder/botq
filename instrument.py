import serial
ser = serial.Serial('/dev/ttyusb0', 115200, timeout=1)
ser.write(b'*IDN?\n')
reply = ser.readline()
reply_str = reply.decode('ascii').strip()


import serial
ser = serial.Serial('/dev/ttyusbo', 115200, timeout=1)
ser.write(b'*IDN?\n')
reply = ser.readline()
reply_str = reply.decode('ascii').strip()