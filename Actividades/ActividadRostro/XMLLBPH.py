import cv2 as cv 
import numpy as np 
import os

dataSet = r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\ActividadRostro\images'
faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataSet+'/'+face
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath+'/'+faceName,0))
    label = label + 1
#print(np.count_nonzero(np.array(labels)==0)) 
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write(r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\ActividadRostro\xlms\LBPH.xml')
