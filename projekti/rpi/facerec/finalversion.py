import cv2
import numpy as np
import os
#from gw_utility.logging import Logging

recognize = cv2.face.LBPHFaceRecognizer_create()
recognize.read('training/finaltrain.yml')
cPath = "dataset/haarcascade_frontalface_default.xml"
fCascade = cv2.CascadeClassifier(cPath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

henkilo = ['None','','','','','','','','','','','','','','','Mikko','Mikko']#array out of bound IndexError

camera = cv2.VideoCapture(0)
camera.set(3, 640) # W
camera.set(4, 480) # H

minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = fCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
        )

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)
        id,confidence = recognize.predict(gray[y:y+h,x:x+w])
        #confidenssi alle 100? 0=best match
        try: #Ugly fix for IndexError but it works
            if(confidence < 100):
                id = henkilo[id]
                confidence = " {0}%".format(round(100-confidence))
            else:
                id = "Tuntematon"
                confidence = " {0}%".format(round(100-confidence))
        except IndexError as e:
            print("hupsista saatana")

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255),2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        cv2.imshow('kamera',img)
    k = cv2.waitKey(1) & 0xFF == ord('q')
    if k == 27:
        break
print("\n [i] Exiting gracefully and doing cleanup")
camera.release()
cv2.destroyAllWindows()
