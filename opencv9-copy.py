import cv2
import numpy as np

faceCascade=cv2.CascadeClassifier('Resources\haarcascade_frontalface_default.xml')
img = cv2.imread("Resources/BTS.jpg")
#img = cv2.resize(img,(int(img.shape[1]/1.5),int(img.shape[0]/1.5)))

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


faces=faceCascade.detectMultiScale(imgGray,1.05,5)
print(faces)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#print (np.shape(img))
#imgGray=cv2.cvtColor(imgGray,cv2.COLOR_GRAY2RGB)

#roi=img[120:260,110:270].copy()
#roiGray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
#roiGray=cv2.cvtColor(roiGray,cv2.COLOR_GRAY2BGR)
#roi=img[120:260,110:270]=roiGray

cv2.imshow('Lena',img)
#cv2.imshow('Lena Gray',imgGray)
#cv2.imshow('Lena ROI',roi)
#cv2.imshow('Lena ROI GRAY',roiGray)
cv2.waitKey(0)