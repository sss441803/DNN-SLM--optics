import serial
import time

# Change the port name according to the need
serialPort = serial.Serial(port = "COM4", baudrate=9600,
                           bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

def fan_on():
    while True:
        serialPort.write(b"2")
        print('on')
        time.sleep(1)
        serialPort.write(b"3")
        print('off')
        time.sleep(6)

def fan_off():
    serialPort.write(b"3")

def laser_on():
    serialPort.write(b"1")

def laser_off():
    serialPort.write(b"0")

def heat_on():
    serialPort.write(b"4")

def heat_off():
    serialPort.write(b"5")