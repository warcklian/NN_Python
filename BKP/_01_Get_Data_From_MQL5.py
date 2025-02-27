import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time

# Conectar con MetaTrader 5
if not mt5.initialize():
    print("Error al inicializar MT5")
    mt5.shutdown()

# Definir el símbolo y el período de tiempo
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M5  # Velas de 5 minutos
n_candles = 10000  # Número de velas a obtener

# Obtener datos históricos
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)

# Convertir a DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Cerrar conexión con MT5
mt5.shutdown()

# Mostrar datos
print(df.head())

# Guardar los datos en CSV para su análisis
df.to_csv("historical_data.csv", index=False)
