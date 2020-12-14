import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset/path/to/pictures'

recognize = cv2.face.LBPHFaceRecognizer_create()
detect = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

def dostuff(path):
    imagePaths = [os.path.join(path,i) for i in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        PIL_image = Image.open(imagePath).convert('L') #greyscale
        image_numpy = np.array(PIL_image,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detect.detectMultiScale(image_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(image_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print("training in process")
faces,ids= dostuff(path)
recognize.train(faces, np.array(ids))
recognize.write('training/namethis.yml')
print("treenit treenattu")
