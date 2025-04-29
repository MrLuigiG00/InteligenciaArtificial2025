import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

# Índices de puntos clave del rostro
selected_points = [33, 133,  # ojo izquierdo
                   362, 263, # ojo derecho
                   4,        # punta de nariz
                   61, 291,  # extremos boca
                   50, 280,  # mejillas
                   10,        # frente
                   1, 0, 12, 14, 17, # nariz y boca
                   152,           # mentón
                   105,            # ceja izquierda
                   159,            # párpado izquierdo
                   9, 168, 20,     # frente-nariz
                   334, 386,       # ceja derecha
                   72, 85,         # labio izquierdo
                   302, 315        # labio derecho
                  ]

def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def distancia_relativa(p1, p2, referencia):
    return distancia(p1, p2) / referencia if referencia != 0 else 0

def linea(p1, p2):
    return cv2.line(frame, p1, p2, (255, 0, 0), thickness=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            puntos = {}

            for idx in selected_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos[idx] = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if 33 in puntos and 362 in puntos:
                # Dibujo de líneas
                if all(k in puntos for k in [9, 4, 105, 159, 334, 386, 133, 61, 362, 291, 0, 17, 72, 85, 302, 315]):
                    linea(puntos[9], puntos[4])  # Frente-nariz
                    linea(puntos[105], puntos[159])  # Ceja izquierda
                    linea(puntos[334], puntos[386])  # Ceja derecha
                    linea(puntos[133], puntos[61])  # Ojo izq - boca
                    linea(puntos[362], puntos[291])  # Ojo der - boca
                    linea(puntos[0], puntos[17])  # Labios externos
                    linea(puntos[61], puntos[291])  # Atravesar boca
                    linea(puntos[72], puntos[85])  # Labio izq interno
                    linea(puntos[302], puntos[315])  # Labio der interno

                    # Ancho del rostro (entre ojos) como referencia
                    ancho_rostro = distancia(puntos[33], puntos[362])

                    # Distancias relativas
                    dist_ceja_izq = distancia_relativa(puntos[105], puntos[159], ancho_rostro)
                    dist_ceja_der = distancia_relativa(puntos[334], puntos[386], ancho_rostro)
                    dist_boca_largo = distancia_relativa(puntos[61], puntos[291], ancho_rostro)
                    dist_boca_alto = distancia_relativa(puntos[0], puntos[17], ancho_rostro)

                    # Umbrales relativos
                    var_enojo = 0.28
                    var_boca = 0.50

                    # Estado emocional estimado
                    if dist_ceja_izq < var_enojo and dist_ceja_der < var_enojo:
                        estado = "Enojado"
                        color = (255, 0, 0)
                    elif dist_boca_alto > var_boca and dist_boca_largo > var_boca and dist_ceja_der > var_enojo and dist_ceja_izq > var_enojo:
                        estado = "Sorprendido"
                        color = (0, 0, 255)
                    else:
                        estado = "Neutral"
                        color = (0, 255, 0)

                    # Mostrar resultados
                    cv2.putText(frame, f"Estado: {estado}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Ceja Izq: {dist_ceja_izq:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Ceja Der: {dist_ceja_der:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Boca Alto: {dist_boca_alto:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Boca Largo: {dist_boca_largo:.2f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('PuntosFacialesMediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
