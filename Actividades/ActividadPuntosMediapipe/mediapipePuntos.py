import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Intentar capturar video hasta que se logre
cap = cv2.VideoCapture(0)

# Lista de índices de landmarks específicos (ojos y boca)
selected_points = [33, 133,  # ojo izquierdo
                   362, 263, # ojo derecho
                   4,        # punto de la nariz
                  61, 291,  # boca
                  50, 280,  # mejillas
                  10,        # frente
                  1, 0, 12, 14, 17, # nariz boca
                  152,           # menton
                  105,            # ceja izquierda
                  159,            #parpado
                  9, 168, 20,   # frente-nariz
                  334, 386,     # ceja derecha
                  72, 85, # labio izquierdo
                  302, 315 # labio derecho
                  ]  


def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def linea(p1, p2):
    """Dibuja una línea entre dos puntos en el frame."""
    return cv2.line(frame, p1, p2, (255, 0, 0), thickness=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            puntos = {}
            
            for idx in selected_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos[idx] = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Dibuja el punto en verde
            
            # Calcular y mostrar distancia entre puntos (ejemplo: entre ojos)
            if 33 in puntos and 133 in puntos:
                #Frente
                linea(puntos[9],puntos[4])
                #Ceja izquierda
                linea(puntos[105],puntos[159])
                #Ceja derecha
                linea(puntos[334],puntos[386])
                #Ojo izquierdo boca
                linea(puntos[133],puntos[61])
                #Ojo derecho boca
                linea(puntos[362],puntos[291])
                #Labios extenos
                linea(puntos[0], puntos[17])
                #Labios internos izquierda
                linea(puntos[72], puntos[85])
                
                #Atravezar boca
                linea(puntos[61], puntos[291])
                
                #Labios internos derecha
                linea(puntos[302], puntos[315])
                
                linea(puntos[50],)
                            
                if 105 in puntos and 159 in puntos and 334 in puntos and 386 in puntos:
                    distancia_ceja_izquierda = distancia(puntos[105], puntos[159])
                    distancia_ceja_derecha = distancia(puntos[334], puntos[386])
                    ancho_rostro = distancia(puntos[33], puntos[362])
                    
                    distancia_boca_largo = distancia(puntos[61], puntos[291])
                    distancia_boca_alto = distancia(puntos[0], puntos[17])
                   
                    var_enojo = 35
                    var_boca = 40
                    
                   
                    
                    if distancia_ceja_izquierda < var_enojo and distancia_ceja_derecha < var_enojo:
                        estado = "Enojado"
                        color = (255, 0, 0)  
                    if distancia_boca_alto > var_boca and distancia_boca_largo > var_boca and distancia_ceja_derecha > var_enojo and distancia_ceja_izquierda > var_enojo:
                        estado = "Sorprendido"
                        color = (0,0,255)
                    else:
                        estado = "Neutral"
                        color = (0, 255, 0)  
                    
                    
    

                        
                     
                    
                    # Dibujar texto en la pantalla
                    cv2.putText(frame, f"Estado: {estado}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) 
                    cv2.putText(frame, f"{distancia_ceja_izquierda:.2f}", (105, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
                
    cv2.imshow('PuntosFacialesMediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()