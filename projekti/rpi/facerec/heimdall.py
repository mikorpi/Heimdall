#Heimdall kasvojentunnistusohjelma
import cv2
import numpy as np
import os

recognize = cv2.face.LBPHFaceRecognizer_create()
recognize.read('training/mikko500.yml')
cPath = "dataset/haarcascade_frontalface_default.xml"
fCascade = cv2.CascadeClassifier(cPath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

persons = ['None','Mikko']

camera = cv2.VideoCapture(0)
camera.set(3, 640) # W
camera.set(4, 480) # H

#mW = 0.1*camera.get(3)
#mH = 0.1*camera.get(4)

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#
    faces = fCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30)
        #(int(mW), int(mH)), enabloi nämä ja ylemmät variablet (line19 & line20) jos haluat käyttää niitä minSize arvojen sijaan.
        )
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)
        #loss = kuinka luotettavasti tunnistaa naaman opetetuksi käyttäjäksi. 0 paras.
        id,loss = recognize.predict(gray[y:y+h,x:x+w])
        #Poikkeuksienhallinta turha,korjattu jo. Koodi pahoitti mielensä line:13 array:sta. Jätän sen kumminkin tänne.
        try:
            if(loss < 100):
                id = persons[1]
                loss = " {0}%".format(round(100-loss))
            else:
                id = "Tuntematon"
                loss = " {0}%".format(round(100-loss))
        except IndexError as e:
            print("indexerror!!!")
#Laitetaan etiketit: NIMI,loss:in määrä, ja applikaation nimi.
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,0,255),2)
        cv2.putText(img, str(loss), (x+5,y+h-5), font, 1, (255,191,0), 1)
        cv2.imshow('Heimdall',img)
        #Ohjelma sulkeutuu q-näppäintä painamalla.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("\n[i] q pressed!\n[i] Exiting gracefully and doing cleanup\n")
            quit()
camera.release()
cv2.destroyAllWindows()
