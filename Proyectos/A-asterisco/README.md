# Proyecto 1: Algoritmo A* / A Estrella
Este proyecto codifica en pygame el algoritmo A* para encontrar la ruta mas corta por busqueda informada.

## Codigo
```
import pygame
from queue import PriorityQueue

# Configuraciones iniciales
ANCHO_VENTANA = 600
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

pygame.font.init() 
FUENTE = pygame.font.SysFont("Arial", 10,)  

# Costos
COSTO_HORIZONTAL_VERTICAL = 10
COSTO_DIAGONAL = 14

# Numero de cuadricula cuadrada
FILAS = 10

class Nodo:
    def _init_(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas

        #Costos por casilla
        self.g = float("inf")  
        self.h = 0 
        self.f = float("inf")
        self.padre = None  

    #Obtener la fila y columna del nodo actual
    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == GRIS

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA
    
    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = GRIS

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_camino(self):
        self.color = VERDE

    def hacer_ruta(self):
        self.color = AZUL

    #Funcion para obtener el numero de la casilla
    def obtener_Casilla(self):
        print(f"{self.col} * {FILAS} +  {self.fila} + 1")
        return self.col * FILAS + self.fila+1    

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

        # Numerar las casillas
        texto = FUENTE.render(f"{self.col * self.total_filas + self.fila + 1}", True, (0, 0, 0))
        ventana.blit(texto, (self.x + 5, self.y + 5))

        # Pintar cordenadas
        texto = FUENTE.render(f"{self.get_pos()}", True, (0, 0, 0))
        ventana.blit(texto, (self.x + 35, self.y+5))


def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo()
            nodo._init_(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

#Calcular distancia de inicio a fin
def calcular_h(nodo, fin):
    x1, y1 = nodo.get_pos()
    x2, y2 = fin.get_pos()
    return (abs(x2 - x1) + abs(y2 - y1))*COSTO_HORIZONTAL_VERTICAL


# Verifica alrededor del nodo actual los vecinos y si alguno es pared
def obtener_vecinos(nodo, grid):
    vecinos = []
    print("Coordenadas del nodo: ", nodo.get_pos(), " Casilla: ",nodo.obtener_Casilla())
    if nodo.fila < nodo.total_filas - 1 and not grid[nodo.fila + 1][nodo.col].es_pared():  # Abajo
        vecinos.append(grid[nodo.fila + 1][nodo.col])
    if nodo.fila > 0 and not grid[nodo.fila - 1][nodo.col].es_pared():  # Arriba
        vecinos.append(grid[nodo.fila - 1][nodo.col])
    if nodo.col < nodo.total_filas - 1 and not grid[nodo.fila][nodo.col + 1].es_pared():  # Derecha
        vecinos.append(grid[nodo.fila][nodo.col + 1])
    if nodo.col > 0 and not grid[nodo.fila][nodo.col - 1].es_pared():  # Izquierda
        vecinos.append(grid[nodo.fila][nodo.col - 1])
    if nodo.fila < nodo.total_filas - 1 and nodo.col < nodo.total_filas - 1 and not grid[nodo.fila + 1][nodo.col + 1].es_pared():  # ⤵
        vecinos.append(grid[nodo.fila + 1][nodo.col + 1])
    if nodo.fila < nodo.total_filas - 1 and nodo.col > 0 and not grid[nodo.fila + 1][nodo.col - 1].es_pared():  # ↙
        vecinos.append(grid[nodo.fila + 1][nodo.col - 1])
    if nodo.fila > 0 and nodo.col < nodo.total_filas - 1 and not grid[nodo.fila - 1][nodo.col + 1].es_pared():  # ↗
        vecinos.append(grid[nodo.fila - 1][nodo.col + 1])
    if nodo.fila > 0 and nodo.col > 0 and not grid[nodo.fila - 1][nodo.col - 1].es_pared():  # ↖
        vecinos.append(grid[nodo.fila - 1][nodo.col - 1])
    return vecinos

def reconstruir_camino(nodo_actual):
    camino = []
    camino.append(nodo_actual.get_pos())

    while nodo_actual.padre:
        nodo_actual = nodo_actual.padre
        nodo_actual.hacer_ruta()
        camino.append(nodo_actual.get_pos())
        

    # Imprimir la ruta final
    print("Ruta Final:")
    camino.reverse()
    for pos in camino:
        print(pos)


def A_estrella(inicio, fin, grid):
    #lista QUEUE tiene la estructura PRIORIDAD, ORDEN, CONTENIDO
    open_set = PriorityQueue()
    i = 0
    #Agregamos el nodo inicial a la lista abierta
    open_set.put((0, i, inicio))
    closed_set = set()

    # Inicializamos g y f del inicio
    inicio.g = 0
    inicio.f = calcular_h(inicio, fin)
    

    while not open_set.empty():
        current = open_set.get()[2]
        closed_set.add(current)

        print("Lista de abiertos: ", [n[2].get_pos() for n in open_set.queue])
        print("Lista de cerrados: ", [nodo.get_pos() for nodo in closed_set])

        # cuando el nodo actual este en la posicion del nodo final, reconstruir el camino
        if current == fin:
            reconstruir_camino(fin)
            return True
        # De los vecinos verificar si esta en el arreglo de vecinos (siendo casilla disponible o pared)
        for vecino in obtener_vecinos(current, grid):
            if vecino in closed_set:
                continue

            peso = (
                        COSTO_DIAGONAL
                        if calcular_h(vecino, current) == 2
                        else COSTO_HORIZONTAL_VERTICAL
                    )
            
            print("Peso de  ", vecino.get_pos(), ":", peso)

            temp_g = current.g + peso
            # Verificar si me combiene cambiar de G
            if temp_g < vecino.g:
                vecino.padre = current
                vecino.g = temp_g
                vecino.h = calcular_h(vecino, fin)
                vecino.f = vecino.g + vecino.h

                if vecino not in open_set.queue:
                    i += 1
                    open_set.put((vecino.f, i, vecino))
                    if(vecino != fin):
                        vecino.hacer_camino()
        
        if current != inicio:
            current.hacer_camino()
        dibujar(VENTANA, grid, FILAS, ANCHO_VENTANA)
        inicio.hacer_inicio()
    return False

def main(ventana, ancho):
    grid = crear_grid(FILAS, ancho)
    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Left click
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()     

                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            elif pygame.key.get_pressed()[pygame.K_r]:
                inicio = None
                fin = None
                grid = crear_grid(FILAS, ancho)
                print("Grid reseteada")

            elif pygame.key.get_pressed()[pygame.K_SPACE]:
                print("--------------------")
                print("Inicio: ", inicio.get_pos())
                print("Nodo: ", nodo.get_pos())
                print("Fin: ", fin.get_pos())
                print(f"H: {calcular_h(inicio, fin)}")
                A_estrella(inicio, fin, grid)
                

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)
```