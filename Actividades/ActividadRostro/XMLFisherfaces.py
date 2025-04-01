import cv2 as cv
import numpy as np
import os

# Ruta del dataset
dataSet = r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\ActividadRostro'
faces = os.listdir(dataSet)

print(faces)

labels = []
facesData = []
label = 0

# Recorremos todas las carpetas dentro del directorio principal
for face in faces:
    facePath = os.path.join(dataSet, face)

    # Verificamos que sea un directorio antes de continuar
    if os.path.isdir(facePath):  
        # Recorremos los archivos dentro del directorio de cada persona
        for faceName in os.listdir(facePath):
            faceImgPath = os.path.join(facePath, faceName)

            # Verificamos que sea un archivo de imagen válido (jpg, png, etc.)
            if os.path.isfile(faceImgPath) and faceName.endswith(('.jpg', '.png', '.jpeg')):
                print(f"Cargando imagen: {faceImgPath}")
                img = cv.imread(faceImgPath, 0)  # Cargar en escala de grises

                if img is None:
                    print(f"🚨 No se pudo leer la imagen: {faceImgPath}")
                    continue

                labels.append(label)
                facesData.append(img)

        label += 1  # Aumentamos el valor de la etiqueta para la siguiente persona

# Verificamos que haya al menos dos clases para entrenar
if len(labels) < 2:
    print("🚨 Se requieren al menos 2 clases para entrenar el modelo.")
else:
    # Crear y entrenar el reconocedor de rostros
    faceRecognizer = cv.face.FisherFaceRecognizer_create()
    faceRecognizer.train(facesData, np.array(labels))

    # Guardar el modelo entrenado
    faceRecognizer.write('laloFisherFace.xml')
    print("Modelo entrenado y guardado como laloFisherFace.xml")
