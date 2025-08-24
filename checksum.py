from pathlib import Path

usb_ports = list(Path('/dev').glob('ttyUSB*'))
port_names = [str(p) for p in usb_ports]
print(port_names)

import hashlib
h = hashlib.sha256()
with open('transfer.bin', 'rb') as f:
    h.update(f.read())