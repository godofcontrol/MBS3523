import cv2
print(cv2.__version__)

img = cv2.imread("Resources/lena.png")
cv2.imshow('Lena',img)
cv2.waitKey(1000)