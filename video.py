import cv2
print(cv2.__version__)

# img = cv2.imread("Resources/lena.png")
# cv2.imshow('Lena',img)
# cv2.waitKey(1000)

cap = cv2.VideoCapture('Resources/dog.mp4')
while True:
    success , img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Dog',img)
    cv2.imshow('Dog Gray', imgGray)
    if cv2.waitKey(100)&0xff==ord('q'):
        break

cap.releaase()
cv2.destroyAllWindow()
