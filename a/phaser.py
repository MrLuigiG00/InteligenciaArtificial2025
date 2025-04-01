import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Generar datos artificiales
np.random.seed(0)
X = np.random.rand(1000, 2)  # 1000 puntos con 2 características cada uno
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Etiqueta 1 si la suma de las características > 1, de lo contrario 0

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal multicapa
model = Sequential([
    Dense(4, input_dim=2, activation='relu'),  # Capa oculta con 4 neuronas y activación ReLU
    Dense(1, activation='sigmoid')            # Capa de salida con 1 neurona y activación sigmoide
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en el conjunto de prueba: {accuracy:.2f}")

# Probar con un nuevo dato
nuevo_dato = np.array([[0.8, 0.3]])  # Ejemplo con características específicas
prediccion = model.predict(nuevo_dato)
print(f"Predicción para {nuevo_dato}: {prediccion[0][0]:.2f}")


#LCTM 