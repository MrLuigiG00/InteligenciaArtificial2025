# Modelado de 5 en Raya
### Instrucciones:
*Modelar una red neuronal que pueda jugar al 5 en linea sin gravedad en un tablero de 20 * 20*

- Definir el tipo de red neuronal y desribir cada una de sus partes
    - Convolucional esto con la finalidad de detectar patrones locales de las coordenadas de las fichas y la informacion que utilizaria como informacion de entrada:
        - Dimensiones del area de juego
        - coordenadas x e y para las fichas de ambos jugadores
- Definir los patrones a utlizar
    - Unos de los patrones que se puede utilizar debido a que el chiste es hacer 5 fichas en linea recta, se puede usar Victoria ya que pueden ser diagonales, verticales u horizontales las lineas para ganar.
    - Algun patron que pueda detectar alguna secuencia si el rival esta por ganar o bloquear el paso a las 5 fichas.
- Definir funcion de activacion es necesaria para este problema
    - Como funcion de activacion podria ser softmax para obtener los posibles movimientos a realizar
- ¿Que valores a la salida de la red se podria esperar?
    - Respuestas de 0/1 o verdadero y falso sobre la coordenada donde se coloca la ficha
- ¿Cuales son los valores maximos que puede tener el bias?
    - Supongo que un 0.80 para que no pueda jugar de manera perfecta

    