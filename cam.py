import cv2
import numpy as np
import face_recognition
print(cv2.__version__)
print(face_recognition.__version__)

img_BG = face_recognition.load_image_file('Images_Known/Bill Gates.jpg')
img_BG = cv2.cvtColor(img_BG,cv2.COLOR_RGB2BGR)
BG_Loc=face_recognition.face_locations(img_BG)[0]
BG_encode=face_recognition.face_encodings(img_BG)[0]
print(len(BG_encode))

img_unknown = face_recognition.load_image_file('Images_Unknown/wy-2.png')
img_unknown = cv2.cvtColor(img_unknown,cv2.COLOR_RGB2BGR)
unknown_Loc=face_recognition.face_locations(img_unknown)[0]
unknown_encode=face_recognition.face_encodings(img_unknown)[0]
#print(BG_Loc)
#print(type(BG_Loc))

#unknown face
cap = cv2.videoCapture(0)
cap.set(3,640)
cap.set(6,480)


while True:
    success,img =cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    unknown_Loc = face_recognition.face_locations(imgRGB)[0]
    unknown_encode = face_recognition.face_encodings(imgRGB)[0]

#compare face
    results=face_recognition.compare_faces([BG_encode],unknown_encode,0.5)
    print(results)
    dist=face_recognition.face_distance([BG_encode],unknown_encode)
    print(dist)
    cv2.rectangle(img_BG,(BG_Loc[3],BG_Loc[0]),(BG_Loc[1],BG_Loc[2]),(0,255,0),2)
    cv2.rectangle(img_unknown,(unknown_Loc[3],unknown_Loc[0]),(unknown_Loc[1],unknown_Loc[2]),(0,255,255),2)

    cv2.imshow('Photo unknown',img_unknown)
    cv2.imshow('Photo BG',img_BG)
    cv2.waitKey(0)