# Detección de Rostros con Modelos Haarcascade

En esta práctica se utilizó el clasificador `haarcascade_frontalface_alt.xml` para realizar detección de rostros en videos.  
El objetivo principal fue etiquetar y reconocer rostros de distintas fuentes utilizando un conjunto de aproximadamente **500 muestras**.

## Fuentes de los rostros utilizados

- Fotografías personales
- Elliot
- Alva Majo
- Dross

## Detalles técnicos

Se procesó un video de entrada utilizando OpenCV para identificar rostros en cada cuadro.  
Cada rostro detectado fue recortado, redimensionado a 100x100 píxeles y almacenado como imagen para su posterior análisis o entrenamiento de modelos de reconocimiento facial.

## Recursos utilizados

- Clasificador Haar: `haarcascade_frontalface_alt.xml`
- Biblioteca: OpenCV
- Video fuente: `alva.mp4`
- Salida de imágenes: Carpeta `Alva Majo`

## Proceso de extracción

1. Se lee el video cuadro por cuadro.
2. Se convierte cada cuadro a escala de grises.
3. Se aplica el clasificador Haar para detectar rostros.
4. Por cada rostro detectado:
   - Se recorta y redimensiona a 100x100 píxeles.
   - Se guarda como archivo `.jpg` (una imagen cada 30 cuadros).
5. Se visualiza en tiempo real tanto el video como los rostros recortados.


---

Este procedimiento permite crear un conjunto de datos personalizado con rostros etiquetados, útil para tareas de entrenamiento y validación de modelos de reconocimiento facial.

## Codigo para generar las imagenes etiquedas
```python
import numpy as np
import cv2 as cv
import math 

rostro = cv.CascadeClassifier(r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\Actividad Rostro\haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\Actividad Rostro\alva.mp4')
i = 0  
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
       #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
       frame2 = frame[ y:y+h, x:x+w]
       #frame3 = frame[x+30:x+w-30, y+30:y+h-30]
       frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
       
        
       if(i%30==0):
        cv.imwrite('C:\\Users\\VIANNEY VARGAS\\Documents\\Inteligencia Aritificial\\Proyectos\\InteligenciaArtificial2025\\Actividades\\Actividad Rostro\\Alva Majo\\Alva Majo' + str(i) + '.jpg', frame2)
        cv.imshow('rostror', frame2)
    cv.imshow('rostros', frame)
    i = i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```


## Reconocimiento con Fisherfaces

```python
import cv2 as cv
import os 

faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.read(r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\ActividadRostro\xlms\caras_set.xml')
faces = ["Persona1", "Persona2", "Persona3"]  # Asegúrate de que esta lista esté correctamente poblada
cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier(r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\ActividadRostro\haarcascade_frontalface_alt.xml')
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for(x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2,  (100,100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        cv.putText(frame, '{}'.format(result), (x, y-20), 1, 3.3, (255, 255, 0), 1, cv.LINE_AA)
        
        if result[1] < 500:
            if result[0] < len(faces):  # Verificación del índice
                cv.putText(frame, '{}'.format(faces[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            else:
                cv.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()

```
## Reconocimiento con Eigenface
```python
import cv2 as cv
import os

# Definir rutas de los archivos necesarios
modelo_path = r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\ActividadRostro\xlms\LuisIgnacioEigenface.xml'
cascade_path = r'C:\Users\VIANNEY VARGAS\Documents\Inteligencia Aritificial\Proyectos\InteligenciaArtificial2025\Actividades\ActividadRostro\haarcascade_frontalface_alt.xml'

# Verificar si el archivo del modelo existe
if not os.path.exists(modelo_path):
    print(f"Error: No se encontró el modelo en {modelo_path}")
    exit()

# Cargar modelo de reconocimiento facial
try:
    faceRecognizer = cv.face.EigenFaceRecognizer_create()
    faceRecognizer.read(modelo_path)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Verificar si el archivo del clasificador existe
if not os.path.exists(cascade_path):
    print(f"Error: No se encontró el clasificador en {cascade_path}")
    exit()

# Cargar clasificador de rostros
rostro_cascade = cv.CascadeClassifier(cascade_path)
if rostro_cascade.empty():
    print("Error: No se pudo cargar 'haarcascade_frontalface_alt.xml'")
    exit()

# Diccionario de nombres asociados a etiquetas (ajusta según tu dataset)
faces = {0: "Persona1", 1: "Persona2"}

# Iniciar captura de video
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in rostros:
        rostro_recortado = gray[y:y + h, x:x + w]
        rostro_recortado = cv.resize(rostro_recortado, (100, 100), interpolation=cv.INTER_CUBIC)

        # Verificar si la imagen tiene el tamaño correcto
        if rostro_recortado.shape[:2] == (100, 100):
            try:
                label, confidence = faceRecognizer.predict(rostro_recortado)

                # Determinar si se reconoce el rostro o no
                if confidence < 2800:
                    nombre = faces.get(label, "Desconocido")
                    color = (0, 255, 0)  # Verde
                else:
                    nombre = "Desconocido"
                    color = (0, 0, 255)  # Rojo

                # Dibujar rectángulo y etiqueta
                cv.putText(frame, nombre, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            except Exception as e:
                print(f"Error al predecir: {e}")

    # Mostrar la imagen en una ventana
    cv.imshow('Reconocimiento Facial', frame)

    # Salir con la tecla ESC
    if cv.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()

```

## Reconocimiento con LBPH
```python
import cv2 as cv
import os 

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read('./xmls/caras_set.xml')

cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier('./xmls/haarcascade_frontalface_alt.xml')
faces = ['Alva Majo', 'dross', 'elliot', 'hiram', 'LuisIgnacio']
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for(x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2,  (100,100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (255,255,0), 1, cv.LINE_AA)
        if result[1] < 50:
            cv.putText(frame,'{}'.format(faces[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2) 
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```