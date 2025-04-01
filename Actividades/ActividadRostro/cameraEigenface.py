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
