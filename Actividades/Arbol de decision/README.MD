# Clasificación con Árbol de Decisión usando Scikit-learn

Este proyecto demuestra cómo cargar un conjunto de datos, entrenar un modelo de clasificación basado en árboles de decisión, y visualizar el árbol resultante utilizando la biblioteca graphviz.

## Descripción del Proceso

1. **Carga del Dataset**
   - Se cargar un csv de un minuto de juego en phaser

2. **Separación de Características y Etiquetas**
   - X: Las dos primeras columnas del dataset se utilizan como características.
   - y: La tercera columna representa las etiquetas o clases.

3. **División de Datos**
   - Se divide el dataset en conjuntos de entrenamiento y prueba usando train_test_split, con un 20% de los datos reservados para prueba.

4. **Entrenamiento del Modelo**
   - Se crea un modelo de DecisionTreeClassifier y se entrena con los datos de entrenamiento.

5. **Predicción y Evaluación**
   - Se realiza la predicción con el conjunto de prueba y se imprimen los resultados.

6. **Visualización del Árbol**
   - El modelo entrenado se exporta a formato DOT y se genera un gráfico con graphviz.