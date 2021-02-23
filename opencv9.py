import cv2
import numpy as np
print(cv2.__version__)

img = cv2.imread("Resources/lena.png")
img = cv2.resize(img,(int(img.shape[1]/1.5),int(img.shape[0]/1.5)))

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print (np.shape(img))
imgGray=cv2.cvtColor(imgGray,cv2.COLOR_GRAY2RGB)

roi=img[120:260,110:270].copy()
roiGray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
roiGray=cv2.cvtColor(roiGray,cv2.COLOR_GRAY2BGR)
roi=img[120:260,110:270]=roiGray

cv2.imshow('Lena',img)
cv2.imshow('Lena Gray',imgGray)
cv2.imshow('Lena ROI',roi)
cv2.imshow('Lena ROI GRAY',roiGray)
cv2.waitKey(0)