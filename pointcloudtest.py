import serial
import struct
import time

SERIAL_COMPORT_CTRL = 'COM4'  ## Enhanced Com Port
SERIAL_COMPORT_DATA = 'COM5' ## Standard Com port

import traceback

def every(delay, task):
  next_time = time.time() + delay
  while True:
    time.sleep(max(0, next_time - time.time()))
    try:
      task()
      return
    except Exception:
      traceback.print_exc()
      # in production code you might want to have this instead of course:
      # logger.exception("Problem while executing repetitive task.")
    # skip tasks if we are behind schedule:
    next_time += (time.time() - next_time) // delay * delay + delay
stop = False
def foo():
    global stop
    stop = True
    print("stop")

import threading
th = threading.Thread(target=lambda: every(300, foo))
th.start()


def startRadarModule():
    ser_ctrl = serial.Serial(SERIAL_COMPORT_CTRL, 115200)
    serial_cnt = 0

    fHandle = open('AOP_6m_default.cfg', mode='r', encoding='utf-8')
    line = None
    line = fHandle.readlines()

    for i in line:
        send_data = i.replace("\n", "\r\n")
        ser_ctrl.write(send_data.encode())

        while True:
            byteCount = ser_ctrl.inWaiting()

            s = ser_ctrl.read(byteCount)
            if byteCount != 0:
                serial_cnt += 1
                print(serial_cnt, byteCount, s.decode())
                break

        time.sleep(1)

    print('Sensor Start')
    ser_ctrl.close()

# startRadarModule()

ser = serial.Serial(SERIAL_COMPORT_DATA, 921600)
recvMsg_list = []

print('Ready')
for i in range(3):
    print(i+1)
    time.sleep(1)
print('Start')


tstamp = time.time()
# f = open("./NewRawData/Spread_"+str(tstamp)+".txt", 'w')
f = open("./NewData/230525/raw/Spin_2.txt", 'w')
while not stop:
    if ser.readable():
        for c in ser.read():
            recvMsg_list.append(c)
            print(c)
            f.write('%d ' % c)
f.close()
exit()

