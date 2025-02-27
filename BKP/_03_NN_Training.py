import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Cargar datos preprocesados
df = pd.read_csv("processed_data.csv")

# Preparar datos de entrada y salida
X = []
y = []

# Definir el número de timesteps para la serie temporal
n_steps = 10

for i in range(n_steps, len(df)):
    X.append(df.iloc[i - n_steps:i, 1:].values)  # Usamos las columnas numéricas
    y.append(df.iloc[i, 4])  # Predecimos el precio de cierre

X = np.array(X)
y = np.array(y)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear la red neuronal LSTM
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, X.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
model.save("lstm_model.h5")

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
