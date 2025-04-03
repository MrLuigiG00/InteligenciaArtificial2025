# Modelar una red neuronal que pueda identificar emociones atraves de los valores obtenidos de las landmarks que genere mediapipe

- ### Definir el tipo de red neuronal y describir cada una de sus partes
    R: Por como veo el proceso de en el que me llevo manipular el codigo para obtener las landmarks del rostro y poder identificar los rasgos faciles que se presentan en las diferentes emociones. Podria trabajarse con una CNN al ser identificacion de imagen
    pero el estar procesando esas imagenes como un dataset idefinido podria llevar a ser costoso en rendimiento. Siendo importante el siguiente detalle, no se analizan imagenes, sino valores numericos y coordenadas.
    
    Por lo que una RNA del tipo Multicapa ya que en esta ser la capa de entra que recibe las coordenadas y distancias y le permita aprender a partir de los datos de entrada a identificar los rostros.
    Tenindo la capa de entrada, las capas ocultas densas que realizan el proceso de identificacion de rostro no linealmente y la capa de salida.
    Pero en el caso del codigo que hemos trabajado con mediapipe en emociones es una red neuronal simple sin aprendimiento, lo que hace que siempre en la imagen actual que este captando este dando el valor y no en base a valores anteriores. Pero si es para detectar rostros las la red se basaria en las proporciones guardadas para determinar que persona es a la que esta viendo.

- ### Definir los patrones a utilizar
    R: Los patrones que se utilizan es en base a las distancias entre los diferentes puntos que veamos que son significativos para detectar rostros por ejemplo las distancias de los labios, el entreabierto de los ojos, la distancias de las cejas con respecto a los perpadados.

- ### Definir funcion de activaciones es necesaria para este problema
    R: Se podria utilizar una Softmax que permita darnos como salida el porcentaje de que tan es tal emocion y en base a la de mayor probabilidad definir una respuesta

- ### Definir el numero maximo de entradas
    R: En este numero de entradas es importante definir bien las emociones que se definir y las landmarks mas importantes, este valor seria multiplicado 2 en caso de que trabajemos en 2 dimensiones sin importar la profundidad de donde este la persona, si se toma encuenta seria por 3. En mi ejemplo utilizo aproximadamente 30 landmarks que identifican puntos como el largo de los ojos y boca, distancias de las cejas, distancia de frente a nariz estos siendo puntos en conjunto por que en base a esos puntos se calculan las distancias relativas conforme al rostro.

- ### ¿Que valores a la salida de la red se podrian esperar?
    R: Si lo trabajamos con las emociones los valores de salida seran en base a la expresion que ingresamos como entrada, enojado, neutral, triste, emocionado, etc, pero si lo tomamos como deteccion de rostros en general, esto dependera si lo que se espera es que identifique a las personas por medio de los rasgos fisicos que posee

- ### ¿Cuales son los valores maximos que pueden tener el bias?
    R: Consideran que no existe un valor perfecto, podria optar por manejar un rango de entre 0 y 1 para poder interpretar las respuestas