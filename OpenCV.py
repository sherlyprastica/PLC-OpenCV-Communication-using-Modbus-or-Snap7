#==================Program by=========================
#=============Endrika Decky Marselino=================

#=====import library========
import cv2 as cv
import numpy as np
import snap7
from snap7.util import*
from snap7.types import*
from pymodbus.client.sync import ModbusTcpClient as mbclient

#=============If Use Snap7 Protocol===================
'''
def ReadMemory(plc,byte,bit,datatype):
    result = plc.read_area(areas['MK'],0,byte, datatype)
    if datatype==S7WLBit:
        return get_bool(result,0,1)
    elif datatype==S7WLByte or datatype==S7WLWord:
        return get_int(result,0)
    elif datatype==S7WLReal:
        return get_real(result,0)
    elif datatype==S7WLDWord:
        return get_dword(result,0)
    else:
        return None

def WriteMemory(plc, byte, bit, datatype, value):
    result = plc.read_area(areas['MK'],0,byte, datatype)
    if datatype==S7WLBit:
        set_bool(result,0,bit,value)
    elif datatype==S7WLByte or datatype==S7WLWord:
        set_int(result,0,value)
    elif datatype==S7WLReal:
        set_real(result,0,value)
    elif datatype==S7WLDWord:
        set_dword(result,0,value)
    plc.write_area(areas['MK'],0,byte,result)

IP = '192.168.0.1'

RACK = 0
SLOT = 1

plc = snap7.client.Client()
plc.connect(IP, RACK, SLOT)

state = plc.get_cpu_state()
print(f'State:{state}')
'''
#=====================================================

#============If Use Modbus TCP========================
'''
client = mbclient('192.168.0.1',port=502) #IP PLC
client.connect()
print('Connect')
UNIT = 0x1
'''
#=====================================================

def nothing(x):
    #any operation
    pass

font = cv.FONT_HERSHEY_TRIPLEX

train_color = 0 #variable to train color

train_b_l = 0
train_g_l = 0
train_r_l = 0
train_b_u = 0
train_g_u = 0
train_r_u = 0

#======to make trackbars in windows============
cv.namedWindow("T")
cv.createTrackbar("L-B", "T", 255, 255, nothing)
cv.createTrackbar("L-G", "T", 255, 255, nothing)
cv.createTrackbar("L-R", "T", 255, 255, nothing)
cv.createTrackbar("U-B", "T", 0, 255, nothing)
cv.createTrackbar("U-G", "T", 0, 255, nothing)
cv.createTrackbar("U-R", "T", 0, 255, nothing)

vid = cv.VideoCapture(0) #open camera

while True:
    ret, frame = vid.read() #read camera

    #get data from trackbars
    l_b = cv.getTrackbarPos("L-B", "T")
    l_g = cv.getTrackbarPos("L-G", "T")
    l_r = cv.getTrackbarPos("L-R", "T")
    u_b = cv.getTrackbarPos("U-B", "T")
    u_g = cv.getTrackbarPos("U-G", "T")
    u_r = cv.getTrackbarPos("U-R", "T")

    #rr = client.read_holding_registers(1, 1, unit=UNIT)#read from register 1
    #train_color = rr.registers

    #train_color = ReadMemory(plc, 12, 0, S7WLWord) #data from snap 7 communication in mw12

    print('Train Color', train_color)

    #=====uncoment if use modbus======
    #if train_color == [1]:

    #=====uncoment if use snap7=========
    #if train_color ==1: #to train using snap7 data
        lower = np.array([l_b, l_g, l_r], dtype="uint8")
        upper = np.array([u_b, u_g, u_r], dtype="uint8")
        train_b_l = l_b
        train_g_l = l_g
        train_r_l = l_r
        train_b_u = u_b
        train_g_u = u_g
        train_r_u = u_r
    else:
        lower = np.array([train_b_l, train_g_l, train_r_l], dtype="uint8")
        upper = np.array([train_b_u, train_g_u, train_r_u], dtype="uint8")

    blur = cv.GaussianBlur(frame,(15, 15),0) #make blur image
    thresh = cv.inRange(blur, lower, upper) #convert from rgb image to binary image

    contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #find contour image

    for cnt in contours:
        area = cv.contourArea(cnt) #get area from image
        approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt,True),True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        #=============fuction to draw shape and send data to plc using snap 7 protocol or modbus================
        if area > 700:
            if 4 <= len(approx) < 5:
                cv.drawContours(frame, [approx], 0, (0, 0, 255,), 5)
                cv.putText(frame, "Square", (x, y), font, 1, (255, 255, 0))
            elif 5 <= len(approx) <8:
                cv.drawContours(frame, [approx], 0, (0, 0, 255,), 5)
                cv.putText(frame, "Pentagon", (x, y), font, 1, (255, 255, 0))
            elif 8 <= len(approx) < 10:
                cv.drawContours(frame, [approx], 0, (0, 0, 255,), 5)
                cv.putText(frame, "Circle", (x, y), font, 1, (255, 255, 0))

            print('Camera Data',len(approx))

            #=====uncoment if use modbus======
            #client.write_register(0,len(approx), unit=UNIT) #if using modbus register 0

            #=====uncoment if use snap7=========
            #WriteMemory(plc, 10, 0, S7WLWord, len(approx)) #if using snap7 mw10

    cv.imshow('Blur Img', blur)
    cv.imshow('Binary Img', thresh)
    cv.imshow('Original Img', frame)

    if cv.waitKey(1) == ord('q'):
        break

vid.release()
cv.destroyAllWindows()