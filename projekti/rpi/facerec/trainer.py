import cv2
import numpy as np
from PIL import Image
import os
#polku datalle mitä tullaan hyödyntämään.
path = 'dataset/path/to/pictures'

#Local Binary Patterns Histograms. Haluaa datan grayscalekuvina.
recognize = cv2.face.LBPHFaceRecognizer_create()
detect = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

#funktio joka toteuttaa opettamisen hyödyntäen numpy sekä PIL kirjastoa.
def dostuff(path):
    imagepaths = [os.path.join(path,i) for i in os.listdir(path)]
    facesamples=[]
    ids = []

    for imagepath in imagepaths:
        PIL_img = Image.open(imagepath).convert('L') #greyscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagepath)[-1].split(".")[1])
        faces = detect.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            facesamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return facesamples,ids
#Prosessointi alkaa
print("\n[i] training in process\n")
faces,ids= dostuff(path)
recognize.train(faces, np.array(ids))
recognize.write('training/namethis.yml')
print("\n[i] training done.\n")
