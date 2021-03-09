import cv2
import numpy as np

def nothing(): pass

cv2.namedWindow('Trackbars')

cv2.createTrackbar('HueLow','Trackbars',101,179,nothing)
cv2.createTrackbar('HueHigh','Trackbars',150,179,nothing)
cv2.createTrackbar('SatLow','Trackbars',79,255,nothing)
cv2.createTrackbar('SatHigh','Trackbars',255,255,nothing)
cv2.createTrackbar('ValLow','Trackbars',55,255,nothing)
cv2.createTrackbar('ValHigh','Trackbars',255,255,nothing)


# Set up webcam
# cam = cv2.VideoCapture(1)
# cam.set(3,640)
# cam.set(4,480)

# Start capturing and show frames on window
while True:
    # success, img = cam.read()
    img = cv2.imread('Resources/smarties.png')
    # cv2.moveWindow('Frame', 100,20)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hueLow = cv2.getTrackbarPos('HueLow','Trackbars')
    hueHigh = cv2.getTrackbarPos('HueHigh', 'Trackbars')
    satLow = cv2.getTrackbarPos('SatLow', 'Trackbars')
    satHigh = cv2.getTrackbarPos('SatHigh', 'Trackbars')
    valLow = cv2.getTrackbarPos('ValLow', 'Trackbars')
    valHigh = cv2.getTrackbarPos('ValHigh', 'Trackbars')

    FGmask = cv2.inRange(hsv, (hueLow,satLow,valLow),(hueHigh,satHigh,valHigh))
    cv2.imshow('FGmask',FGmask)

    contours, aaa =cv2.findContours(FGmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)

    cv2.drawContours(img,contours,-1,(255,0,0),3)

    cv2.imshow('Frame', img)
    if cv2.waitKey(1) == 27:
        break

# cam.release()
cv2.destroyAllWindows()