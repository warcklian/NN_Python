from tensorflow import keras
# Cargar modelo entrenado
model = keras.models.load_model("lstm_model.h5")

# Obtener las últimas 10 velas para hacer una predicción
latest_data = np.array([df.iloc[-n_steps:, 1:].values])

# Hacer la predicción
predicted_price = model.predict(latest_data)
print(f"Precio predicho: {predicted_price[0][0]}")
