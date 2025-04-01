from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, Y = iris.data, iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Modelo
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500, random_state=42) #Preparacion del modelo

mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)

print(Y_pred)
print(X_test)

print(f'\nPrecisión en test: {accuracy: 4f}')
