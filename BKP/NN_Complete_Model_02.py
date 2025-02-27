import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm

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
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]  # Lista de activos a operar
open_positions = {}  # Diccionario para rastrear posiciones abiertas

# Inicializar MT5 correctamente
def initialize_mt5():
    if not mt5.initialize():
        print("Error al inicializar MetaTrader 5")
        return False
    return True

def stop_trading():
    global running
    running = False
    print("Se ha detenido la ejecución del programa.")

def get_market_data(symbol, n_candles=200):
    if not initialize_mt5():
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_candles)
    if rates is None:
        print(f"Error al obtener datos del mercado para {symbol}.")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Datos obtenidos correctamente para {symbol}.")
    return df

def create_or_load_model(symbol):
    progress_bar = tqdm(total=100, desc=f'Creando modelo {symbol}', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    model_file = f"lstm_model_{symbol}.keras"
    if os.path.exists(model_file):
        print(f"Cargando modelo existente para {symbol}...")
        return load_model(model_file)
    
    progress_bar.update(10)
    print(f"Creando nuevo modelo para {symbol}...")
    progress_bar.update(20)
    model = Sequential([
        Input(shape=(10, 7)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    progress_bar.update(30)
    model.compile(optimizer='adam', loss='mse')
    progress_bar.update(40)
    model.save(model_file)
    progress_bar.update(100)
    print(f'Progreso: 100% - Predicción completada para {symbol}')
    progress_bar.close()
    return model

def add_technical_indicators(df):
    df['returns'] = df['close'].pct_change()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['returns'].rolling(14).mean() / df['returns'].rolling(14).std()))
    df['volatility'] = df['returns'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

def send_order(symbol, action):
    print(f"Simulación de envío de orden: {action} en {symbol}")

def real_trading():
    print("Iniciando ejecución en modo real...")
    while running:
        for symbol in SYMBOLS:
            print(f"Solicitando datos del mercado para {symbol}...")
            df = get_market_data(symbol, n_candles=200)
            if df is None:
                continue
            df = add_technical_indicators(df)
            scaler = MinMaxScaler()
            features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50']
            df[features] = scaler.fit_transform(df[features])
            X = np.array([df.iloc[-10:][features].values])
            print(f'Iniciando predicción para {symbol}...')
            progress_bar = tqdm(total=100, desc=f'Predicción {symbol}', bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} [{postfix}]')
            progress_bar.update(50)
            print(f'Progreso: 50% - Procesando datos para {symbol}')
            predicted_price = models[symbol].predict(X)[0][0]
            progress_bar.update(100)
            progress_bar.close()
            print(f'Predicción final para {symbol}: {predicted_price:.6f}')
            
            last_close = df.iloc[-1]['close']
            if predicted_price > last_close * 1.002:
                print(f"Señal de COMPRA detectada en {symbol}")
                send_order(symbol, "BUY")
            elif predicted_price < last_close * 0.998:
                print(f"Señal de VENTA detectada en {symbol}")
                send_order(symbol, "SELL")
        time.sleep(5)

print("Cargando modelos para los símbolos...")
models = {}
with strategy.scope():
    for symbol in SYMBOLS:
        models[symbol] = create_or_load_model(symbol)
        print(f"Modelo para {symbol} listo.")

SYMBOLS = list(models.keys())
print(f"Símbolos con modelos cargados: {SYMBOLS}")
print("Modo real listo para operar.")

mode = "1"  # Definir la variable mode antes de usarla
if mode == "1":
    real_trading()
