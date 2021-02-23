import cv2
print(cv2.__version__)

faceCascade=cv2.CascadeClassifier('Resources\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('Resources/guitar.mp4')

while True:
    success , img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.05, 5)
    print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('facecap',img)
    if cv2.waitKey(10)&0xff==ord('q'):
        break

cap.releaase()
cv2.destroyAllWindow()