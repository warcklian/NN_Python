import pandas as pd
# Cargar datos
df = pd.read_csv("historical_data.csv")

# Seleccionar columnas relevantes
df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# Crear nuevas características
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(window=10).std()

# Eliminar valores nulos
df.dropna(inplace=True)

# Normalizar datos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['open', 'high', 'low', 'close', 'tick_volume', 'returns', 'volatility']] = scaler.fit_transform(
    df[['open', 'high', 'low', 'close', 'tick_volume', 'returns', 'volatility']]
)

# Mostrar datos preprocesados
print(df.head())

# Guardar datos preprocesados
df.to_csv("processed_data.csv", index=False)
