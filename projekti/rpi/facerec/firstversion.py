import numpy as np
import cv2

fCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
capture.set(3,640) # W
capture.set(4,480) # H

while True:
    ret, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20,20)
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
capture.release()
cv2.destroyAllWindows()
