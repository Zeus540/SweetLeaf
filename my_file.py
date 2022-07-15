#!/usr/bin/env python
import time
import serial

def get_data():
 ser = serial.Serial(
      port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0 
      baudrate = 115200,
      parity=serial.PARITY_NONE, 
      stopbits=serial.STOPBITS_ONE,
      bytesize=serial.EIGHTBITS, 
 #      timeout=2
 )
 data = ser.readline(50)
 time.sleep(4)
 endata = data.decode("utf-8")
 ser.close()
 print(endata)
 return  endata
