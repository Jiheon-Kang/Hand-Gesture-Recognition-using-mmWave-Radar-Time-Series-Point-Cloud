import serial
import time

SERIAL_COMPORT_CTRL = 'COM4'  ## Enhanced Com Port
SERIAL_COMPORT_DATA = 'COM5' ## Standard Com port

ser_ctrl = serial.Serial(SERIAL_COMPORT_CTRL, 115200)
serial_cnt = 0

fHandle = open('AOP_6m_default.cfg', mode='r', encoding='utf-8')
line = None
line = fHandle.readlines()
print(line)
for i in line:
    send_data = i.replace("\n", "\r\n")

    ser_ctrl.write(send_data.encode())
    print(send_data.encode())
    time.sleep(0.1)

print('Sensor Start')
ser_ctrl.close()