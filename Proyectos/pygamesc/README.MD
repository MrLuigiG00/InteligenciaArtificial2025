# Proyecto 2: Phaser en Python

**Este proyecto se utilizan 3 tipos de modelo de entrenamiento para jugar automaticamente con aprendizaje automatico** 

## RED NEURONAL 
```python
import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical


# ---------------- MODELO DE RED NEURONAL ----------------
# ---------------- RED NEURONAL DE SALTO -----------------
def entrenar_modelo(datos_modelo):
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6]  
    y = datos[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = Sequential([
        Dense(32, input_dim=6, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelo.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Modelo entrenado con precisión: {accuracy:.2f}")
    
    return modelo

def decidir_salto(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo):
    if modelo_entrenado is None:
        print("Modelo no entrenado. No se puede decidir.")
        return False, en_suelo

    distancia_suelo = abs(jugador.x - bala.x)
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    prediccion = modelo_entrenado.predict(entrada, verbose=0)[0][0]

    if prediccion > 0.5 and en_suelo:
        salto = True
        en_suelo = False
        print("Saltar")

    return salto, en_suelo


 
# ---------------- RED NEURONAL DE MOVIMIENTO -----------------
def entrenar_red_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar.")
        return None

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')  
    y = datos[:, 8].astype('int')       # Acción: izquierda, quieto, derecha

    y_categorical = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, input_dim=8, activation='relu'), 
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión del modelo de movimiento: {accuracy:.2f}")
    
    return model

def decidir_movimiento(jugador, bala, modelo_movimiento, salto, bala_suelo):
    if modelo_movimiento is None:
        print("Modelo no entrenado.")
        return jugador.x, 1  # Quieto por defecto

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[
        jugador.x,                     # Posición actual del jugador
        jugador.y,                     # Altura del jugador
        bala.centerx,                  # X de la bala aérea
        bala.centery,                  # Y de la bala aérea
        bala_suelo.x,                  # X de la bala del suelo
        bala_suelo.y,                  # Y de la bala del suelo
        distancia_bala_suelo,          # Distancia a bala del suelo
        1 if salto else 0              # Si el jugador está saltando
    ]], dtype='float32')

    prediccion = modelo_movimiento.predict(entrada, verbose=0)[0]
    accion = np.argmax(prediccion)

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
        print("Izquierda")
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha")
    else:
        print("Quieto")

    return jugador.x, accion

```

## ARBOL DE DECISIÓN
```python
import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------- ARBOL DE DECISION ----------------

def entrenar_arbol_salto(datos_modelo):
    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar el árbol.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6] 
    y = datos[:, 6]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    arbol = DecisionTreeClassifier(max_depth=10)
    arbol.fit(X_train, y_train)

    accuracy = arbol.score(X_test, y_test)
    print(f"Precisión del árbol de salto: {accuracy:.2f}")

    return arbol

def decidir_salto_arbol(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, arbol_salto, salto, en_suelo):
    if arbol_salto is None:
        print("Árbol no entrenado.")
        return salto, en_suelo

    distancia_suelo = abs(jugador.x - bala.x)
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    prediccion = arbol_salto.predict(entrada)[0]

    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False
        print("Saltar")

    return salto, en_suelo



def entrenar_arbol_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar el árbol de movimiento.")
        return None

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')
    y = datos[:, 8].astype('int')  # 0=izquierda, 1=quieto, 2=derecha

    arbol = DecisionTreeClassifier(max_depth=10)
    arbol.fit(X, y)

    return arbol



def decidir_movimiento_arbol(jugador, bala_aire, arbol_movimiento, salto, bala_suelo):
    if arbol_movimiento is None:
        print("Árbol de movimiento no entrenado.")
        return jugador.x, 1  # Quieto

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[
        jugador.x,
        jugador.y,
        bala_aire.centerx,
        bala_aire.centery,
        bala_suelo.x,
        bala_suelo.y,
        distancia_bala_suelo,
        1 if salto else 0
    ]], dtype='float32')

    accion = arbol_movimiento.predict(entrada)[0]

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
        print("Izquierda")
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha (árbol)")
    else:
        print("Quieto (árbol)")

    return jugador.x, accion
```


## K NEAREST NEIGHBOURS
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

# ----------- KNN PARA DECIDIR SALTO ------------
def entrenar_knn_salto(datos_modelo):
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de salto.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6]  
    y = datos[:, 6]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_knn = KNeighborsClassifier(n_neighbors=10)  
    modelo_knn.fit(X_train, y_train)

    accuracy = modelo_knn.score(X_test, y_test)
    print(f"Precisión del modelo KNN de salto: {accuracy:.2f}")
    
    return modelo_knn

def decidir_salto_knn(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, modelo_knn, salto, en_suelo):
    if modelo_knn is None:
        print("Modelo KNN no entrenado. No se puede decidir.")
        return False, en_suelo

    distancia_suelo = abs(jugador.x - bala.x)
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    prediccion = modelo_knn.predict(entrada)[0]

    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False
        print("Saltar (KNN)")

    return salto, en_suelo

def entrenar_knn_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de movimiento.")
        return jugador.x, 1

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')  
    y = datos[:, 8].astype('int')       # 0: izquierda, 1: quieto, 2: derecha

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_knn = KNeighborsClassifier(n_neighbors=10)
    modelo_knn.fit(X_train, y_train)

    accuracy = modelo_knn.score(X_test, y_test)
    print(f"Precisión del modelo KNN de movimiento: {accuracy:.2f}")

    return modelo_knn

def decidir_movimiento_knn(jugador, bala_aire, modelo_knn_mov, salto, bala_suelo):
    if modelo_knn_mov is None:
        print("Modelo KNN de movimiento no entrenado.")
        return None

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[jugador.x,
                         jugador.y,
                         bala_aire.centerx,
                         bala_aire.centery,
                         bala_suelo.x,
                         bala_suelo.y,
                         distancia_bala_suelo,
                         1 if salto else 0
                         ]], dtype='float32')

    prediccion = modelo_knn_mov.predict(entrada)
    accion = int(prediccion[0])

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
        print("Izquierda (KNN)")
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha (KNN)")
    else:
        print("Quieto (KNN)")

    return jugador.x, accion
```

## Game
```python
import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from modelo_red import *
from arbol_decision import *
from knn import * 

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala_suelo, Bala_aire, Salto, bowser y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)
AZUL = (0, 0, 255)

# Variables del jugador, balas, bowser, fondo, etc.
jugador = None
bala_suelo = None
bala_aire = None
fondo = None
bowser = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True
subiendo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático
modo_arbol = False  # Indica si el modo de juego es árbol de decisión
modo_knn = False
# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []
modelo_entrenado = None

modelo_entrenado_arbol = None
movimiento_entrenado_arbol = None

datos_movimiento = []
modelo_entrenado_movimiento = []



intervalo_decidir_salto = 1  # Ejecutar decidir_salto cada 10 frames
contador_decidir_salto = 0

# Cargar imágenes
bala_img = pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\ball.png'), (44, 44))
bala_aire_img = pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\Flyguy\B0.png'), (30, 30))  # Imagen diferente para bala aérea

fondo_img = pygame.image.load(r'Proyectos\pygamesc\assets\game\fondo2.png')
bowser_img = pygame.image.load(r'Proyectos\pygamesc\assets\game\bowser.png')
shyguy_img = pygame.image.load(r'Proyectos\pygamesc\assets\sprites\Flyguy\F0.png')
menu_img = pygame.image.load(r'Proyectos\pygamesc\assets\game\menu.png')

bloque_img  = pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\game\Block.jpg'), (22, 22))


fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear los rectángulos del jugador, balas y bowser
jugador = pygame.Rect(50, h - 100, 32, 48)
bala_suelo = pygame.Rect(w - 50, h - 90, 16, 16)
bala_aire = pygame.Rect(0, -50, 16, 16)  # Comienza arriba de la pantalla
bowser = pygame.Rect(w - 100, h - 130, 64, 64)
shyguy = pygame.Rect(0, 0, 64, 64)  # Rectángulo para el enemigo

# Variables para el movimiento en zigzag y disparos del Shy Guy
zigzag_direccion = 1  # 1 para derecha, -1 para izquierda
zigzag_velocidad = 5  # Velocidad del movimiento en zigzag
shyguy_disparo_cooldown = 0
shyguy_disparo_intervalo = 60  # frames entre disparos
velocidad_bala_aire = [0, 5]  # [velocidad_x, velocidad_y]

menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para las balas
velocidad_bala_suelo = -10  # Velocidad de la bala_suelo hacia la izquierda
bala_disparada_suelo = False
bala_disparada_aire = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# Temporizador para bala aérea
ultimo_disparo_aire = 0

#Cronometro
tiempo_juego = 0
cronometro_pausado = False
inicio_tiempo = pygame.time.get_ticks()

# ---------------- FUNCIONES DEL SHY GUY ----------------
def mover_shyguy():
    global shyguy, zigzag_direccion, shyguy_disparo_cooldown
    shyguy.x += zigzag_direccion * zigzag_velocidad
    shyguy_disparo_cooldown -= 1
    
    # Cambiar dirección al alcanzar los límites
    if shyguy.x <= 0 or shyguy.x >= 200 - shyguy.width:
        zigzag_direccion *= -1

# Movimimiento del Fly Guy: Quieto
""" def mover_shyguy():
    global shyguy, shyguy_disparo_cooldown
    shyguy.x = (200 - shyguy.width) // 2
    shyguy_disparo_cooldown -= 1 """


def disparar_bala_aire():
    global bala_aire, bala_disparada_aire, velocidad_bala_aire, ultimo_disparo_aire, shyguy_disparo_cooldown
    
    # Solo disparar si el cooldown ha terminado y el Shy Guy está en pantalla
    if not bala_disparada_aire and shyguy_disparo_cooldown <= 0 and 0 <= shyguy.x <= w:
        # La bala comienza en la posición del Shy Guy
        bala_aire.x = shyguy.x + shyguy.width // 2 - bala_aire.width // 2
        bala_aire.y = shyguy.y + shyguy.height
        
        # Velocidad con componente vertical fija para disparo en línea recta
        velocidad_bala_aire[0] = 0  # Sin componente horizontal
        velocidad_bala_aire[1] = 5  # Velocidad vertical constante
        
        bala_disparada_aire = True
        shyguy_disparo_cooldown = shyguy_disparo_intervalo
        ultimo_disparo_aire = pygame.time.get_ticks()

# ---------------- FUNCIONES DEL JUEGO ----------------
def disparar_bala():
    global bala_disparada_suelo, velocidad_bala_suelo
    if not bala_disparada_suelo:
        velocidad_bala_suelo = random.randint(-8, -3)
        bala_disparada_suelo = True

def mover_jugador():
    global jugador, en_suelo, salto, pos_actual
    keys = pygame.key.get_pressed()
    pos_actual = 1  # Posición inicial en el centro
    # Limitar el movimiento del jugador al rango de 0 a 200
    if keys[pygame.K_LEFT] and jugador.x > 0:
        jugador.x -= 5
        pos_actual = 0
    if keys[pygame.K_RIGHT] and jugador.x < 200 - jugador.width:
        jugador.x += 5
        pos_actual = 2
    if keys[pygame.K_UP] and en_suelo:
        salto = True
        en_suelo = False
    
    distancia_x = (jugador.centerx - bala_aire.centerx)
    distancia_y = (jugador.centery - bala_aire.centery)
    distancia_total = (distancia_x**2 + distancia_y**2) ** 0.5
    
    """ if not modo_auto or not modo_arbol or not modo_knn:
        print(f"| Datos: {len(datos_modelo)} | Posicion Actual : {jugador.x} |  Posicion de la bala {bala_aire.centerx} | Distancia horizontal aire(x): {distancia_x} | Distancia vertical aire(y): {distancia_y} | Velocidad Bala Aire: {velocidad_bala_aire} ", end="\r")  """ 
    
def mostrar_cronometro():
    minutos = tiempo_juego // 60000
    segundos = (tiempo_juego % 60000) // 1000
    texto = fuente.render(f"Tiempo: {minutos:02d}:{segundos:02d}", True, BLANCO)
    pantalla.blit(texto, (10, 10))
    
def iniciar_cronometro():
    global inicio_tiempo, cronometro_activo, tiempo_juego
    inicio_tiempo = pygame.time.get_ticks()
    cronometro_activo = True
    tiempo_juego = 0

def actualizar_cronometro():
    global tiempo_juego, inicio_tiempo
    tiempo_juego = pygame.time.get_ticks() - inicio_tiempo

def reset_bala():
    global bala_suelo, bala_disparada_suelo
    bala_suelo.x = w - 50
    bala_disparada_suelo = False
    
def reset_bala_aire():
    global bala_aire, bala_disparada_aire
    bala_aire.y = -50
    bala_disparada_aire = False    

def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo, subiendo

    if salto:
        if subiendo:
            jugador.y -= salto_altura
            salto_altura -= gravedad

            if salto_altura <= 0:
                subiendo = False
        else:
            jugador.y += salto_altura
            salto_altura += gravedad

            if jugador.y >= h - 100:
                jugador.y = h - 100
                salto = False
                salto_altura = 15
                subiendo = True
                en_suelo = True

def update():
    global bala_suelo, bala_aire, current_frame, frame_count, fondo_x1, fondo_x2
    mover_shyguy()
    
    # Cargar las imágenes
    jugador_framesM = [
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M0.png'), (44, 55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M1.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M2.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M3.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M4.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M5.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M6.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M7.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M8.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M9.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M10.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M11.png'),(44,55)),
    ]

    jugador_frames_saltoM =[
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M12.png'), (44, 55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\M\M13.png'), (44, 55)),
    ]

    jugador_framesL = [
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L0.png'), (44, 55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L1.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L2.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L3.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L4.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L5.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L6.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L7.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L8.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L9.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L10.png'),(44,55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L11.png'),(44,55)),
    ]

    jugador_frames_saltoL =[
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L14.png'), (50, 55)),
        pygame.transform.scale(pygame.image.load(r'Proyectos\pygamesc\assets\sprites\L\L12.png'), (50, 55)),
    ]

    if not modo_auto and not modo_knn:
        jugador_framesM = jugador_framesL
        jugador_frames_saltoM = jugador_frames_saltoL

    # Mover el fondo
    fondo_x1 -= 3
    fondo_x2 -= 3

    if fondo_x1 <= -w:
        fondo_x1 = w
    if fondo_x2 <= -w:
        fondo_x2 = w

    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    if salto:
        if subiendo:
            pantalla.blit(jugador_frames_saltoM[0], (jugador.x, jugador.y))
        else:
            pantalla.blit(jugador_frames_saltoM[1], (jugador.x, jugador.y))
    else:
        frame_count += 10
        if frame_count >= frame_speed:
            current_frame = (current_frame + 1) % len(jugador_framesM)
            frame_count = 0
        pantalla.blit(jugador_framesM[current_frame], (jugador.x, jugador.y))

    pantalla.blit(bowser_img, (bowser.x, bowser.y))
    pantalla.blit(shyguy_img, (shyguy.x, shyguy.y+75))
    pantalla.blit(bloque_img, (200, h-50))

    # Mover y dibujar balas
    if bala_disparada_suelo:
        bala_suelo.x += velocidad_bala_suelo
        pantalla.blit(bala_img, (bala_suelo.x, bala_suelo.y))
        
    if bala_disparada_aire:
        bala_aire.x += velocidad_bala_aire[0]  # Componente horizontal
        bala_aire.y += velocidad_bala_aire[1]  # Componente vertical
        pantalla.blit(bala_aire_img, (bala_aire.x, bala_aire.y))

    # Reiniciar balas cuando salen de pantalla
    if bala_suelo.x < 0:
        reset_bala()
        
    if bala_aire.y > h or bala_aire.x < 0 or bala_aire.x > w:
        reset_bala_aire()

    # Detectar colisiones
    
    if jugador.colliderect(bala_suelo) or  jugador.colliderect(bala_aire): 
        print("Colisión detectada!")
        reiniciar_juego()

def guardar_datos():
    global jugador, bala_suelo, velocidad_bala_suelo, salto
    distancia_suelo = abs(jugador.x - bala_suelo.x)
    salto_hecho = 1 if salto else 0

    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    #Guardar datos de salto
    datos_modelo.append((
        velocidad_bala_suelo,
        distancia_suelo,
        distancia_aire_x,
        distancia_aire_y,
        hay_bala_aire,
        jugador.x,
        salto_hecho
    ))

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)
    #Guardar datos de movimiento
    datos_movimiento.append((
        jugador.x,
        jugador.y,
        bala_aire.centerx,
        bala_aire.centery,
        bala_suelo.x,
        bala_suelo.y,
        distancia_bala_suelo,
        1 if salto else 0,
        pos_actual
    ))

def pausa_juego():
    global pausa, cronometro_pausado, inicio_tiempo
    pausa = not pausa
    cronometro_pausado = pausa
    if pausa:
        imprimir_datos()
    else:
        # Ajustar tiempo de inicio al salir de pausa
        inicio_tiempo = pygame.time.get_ticks() - (tiempo_juego * 1000)
        print("Juego reanudado.")

def mostrar_menu():
    global menu_activo, modo_auto, modo_arbol, modo_knn
    global datos_modelo, datos_movimiento
    global modelo_entrenado, modelo_entrenado_movimiento
    global modelo_entrenado_arbol, movimiento_entrenado_arbol

    pantalla.fill(NEGRO)
    actualizar_cronometro()
    mostrar_cronometro()
    texto = fuente.render("'A'uto 'M'anual 'E'ntrenar 'T'ree 'K' Nearest 'Q' para Salir", True, BLANCO)
    pantalla.blit(texto, (w // 4, h // 2))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    if(datos_modelo and datos_movimiento):
                        modo_auto = True
                        modo_arbol = False
                        modo_knn = False
                        menu_activo = False
                        iniciar_cronometro()
                    else:
                        print("Falta jugar para obtener datos, presiona 'M' o si ya jugaste, presiona 'E'")
                if evento.key == pygame.K_e:
                    if len(datos_modelo) > 0 and len(datos_movimiento) > 0:
                        modelo_entrenado = entrenar_modelo(datos_modelo)
                        modelo_entrenado_movimiento = entrenar_red_movimiento(datos_movimiento)

                elif evento.key == pygame.K_m:
                    modo_auto = False
                    modo_arbol = False
                    modo_knn = False
                    menu_activo = False
                    datos_modelo = []
                    datos_movimiento = []
                    iniciar_cronometro()
                elif evento.key == pygame.K_t:
                    modo_auto = False
                    modo_arbol = True
                    modo_knn = False
                    menu_activo = False
                    modelo_entrenado_arbol = entrenar_arbol_salto(datos_modelo)
                    movimiento_entrenado_arbol = entrenar_arbol_movimiento(datos_movimiento)
                    iniciar_cronometro()
                elif evento.key == pygame.K_x:
                    reiniciar_juego()    
                if evento.key == pygame.K_k:
                    modo_auto = False
                    modo_arbol = False
                    modo_knn = True
                    menu_activo = False
                    modelo_entrenado = entrenar_knn_salto(datos_modelo)
                    modelo_entrenado_movimiento = entrenar_knn_movimiento(datos_movimiento)
                    iniciar_cronometro()    
                elif evento.key == pygame.K_q:
                    imprimir_datos()
                    pygame.quit()
                    exit()

def reiniciar_juego():
    global menu_activo, jugador, bala_suelo, bala_aire, bowser, bala_disparada_suelo, bala_disparada_aire, salto, en_suelo
    menu_activo = True
    jugador.x, jugador.y = 50, h - 100
    bala_suelo.x = w - 50
    bala_aire.y = -50
    bowser.x, bowser.y = w - 100, h - 100
    bala_disparada_suelo = False
    bala_disparada_aire = False
    salto = False
    en_suelo = True
    imprimir_datos()
    mostrar_menu()
    
def imprimir_datos():
    for dato in datos_movimiento:
        print(dato)
        
def main():
    global salto, en_suelo, bala_disparada_suelo, bala_disparada_aire, contador_decidir_salto

    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    imprimir_datos()
                    pygame.quit()
                    exit()

        if not pausa:
            # Modo manual
            if not modo_auto and not modo_arbol and not modo_knn:
                mover_jugador()
                if salto:
                    manejar_salto()
                guardar_datos()

            # Modo automático con red neuronal
            if modo_auto and modelo_entrenado and modelo_entrenado_movimiento:
                    salto, en_suelo = decidir_salto(jugador, bala_suelo, velocidad_bala_suelo, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo)
                    manejar_salto()
                    jugador.x, pos_actual = decidir_movimiento(jugador, bala_aire, modelo_entrenado_movimiento, salto, bala_suelo)
                    mover_jugador()
                    
            # Modo K vecinos 
            if modo_knn:         
                    salto, en_suelo = decidir_salto_knn(jugador, bala_suelo, velocidad_bala_suelo, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo)
                    manejar_salto()
                    jugador.x, pos_actual = decidir_movimiento_knn(jugador, bala_aire, modelo_entrenado_movimiento, salto, bala_suelo)

            # Modo con árbol de decisión
            if modo_arbol:         
                    salto, en_suelo = decidir_salto_arbol(jugador, bala_suelo, velocidad_bala_suelo, bala_aire, bala_disparada_aire, modelo_entrenado_arbol, salto, en_suelo)
                    manejar_salto()
                    jugador.x, pos_actual = decidir_movimiento_arbol(jugador, bala_aire, movimiento_entrenado_arbol, salto, bala_suelo)

            # Disparar balas
            if not bala_disparada_suelo:
                disparar_bala()
                
            disparar_bala_aire()  # Intentar disparar bala aérea en cada frame (la función controla el intervalo)
            
            update()
            mostrar_cronometro()
            if not pausa:
                 if not cronometro_pausado:
                    actualizar_cronometro()
        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
```

