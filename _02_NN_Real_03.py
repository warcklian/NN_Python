import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from sklearn.preprocessing import MinMaxScaler
import os

def custom_loss_function(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

# Registrar la función de pérdida personalizada en caso de que los modelos la necesiten
get_custom_objects().update({"custom_loss_function": custom_loss_function})

print("Buscando GPUs disponibles...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detectadas: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
else:
    print("No se detectaron GPUs. Usando CPU.")
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")

running = True
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "USDMXN"]  # Lista de activos a operar
models = {}
open_orders = {}

# Verificar y cargar modelos existentes
def load_models():
    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            try:
                models[symbol] = load_model(model_path, custom_objects={"custom_loss_function": custom_loss_function})
                print(f"Modelo para {symbol} cargado correctamente.")
            except Exception as e:
                print(f"Error al cargar el modelo para {symbol}: {e}")
        else:
            print(f"Modelo no encontrado para {symbol}. Será generado en modo Backtest si es necesario.")

# Obtener datos de mercado con indicadores técnicos
def get_market_data(symbol, n_candles=200):
    print(f"Obteniendo datos de {symbol}...")
    if not mt5.initialize():
        print("Error al inicializar MT5")
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_candles)
    if rates is None:
        print(f"Error al obtener datos del mercado para {symbol}.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / df['close'].diff().apply(lambda x: abs(x)).rolling(window=14).mean()))
    df['volatility'] = df['close'].pct_change().rolling(window=10).std()
    df.dropna(inplace=True)
    return df

# Ejecución del trading en tiempo real
def real_trading():
    print("Iniciando ejecución en modo real...")
    while running:
        for symbol in SYMBOLS:
            df = get_market_data(symbol, n_candles=200)
            if df is None:
                continue
            
            expected_features = models[symbol].input_shape[-1]  # Obtener número de características esperadas
            df_numeric = df[['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']]
            
            if df_numeric.shape[1] != expected_features:
                print(f"Error: el modelo espera {expected_features} características, pero se encontraron {df_numeric.shape[1]} en {symbol}.")
                continue
            
            df_numeric = df_numeric.iloc[-10:]  # Tomar las últimas 10 velas
            X_input = np.array([df_numeric.values.astype(float)])
            predicted_price = models[symbol].predict(X_input)[0][0]
            actual_price = df.iloc[-1]['close']
            print(f"Predicción para {symbol}: {predicted_price}, Precio actual: {actual_price}")
        
        time.sleep(5)

# Cargar modelos al iniciar
load_models()

real_trading()
