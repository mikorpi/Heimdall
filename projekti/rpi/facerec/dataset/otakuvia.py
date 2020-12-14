# Ottaa kuvia n-määrän, määritelty line27 elif-lausekkeessa.
# Line 22 määritä nopeus
import cv2
import os
import time
cam = cv2.VideoCapture(0)

cam.set(3, 640)
cam.set(4, 480)
cc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

c = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cc.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:

        cv2.imwrite("kuvat/final/mikko"+'.'+str(c)+".jpg",gray[y:y+h,x:x+w])
        time.sleep(0.3)
        print("loop",c)
        c+=1
    k = cv2.waitKey(1) & 0xFF == ord ('q')
    if k == 27:
        break
    elif c >= 200:
        break
cam.release()
cv2.destroyAllWindows()
