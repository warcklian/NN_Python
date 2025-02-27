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

def select_mode(mode_choice):
    global mode
    mode = mode_choice
    root.destroy()

# Crear una ventana para la selección del modo
root = tk.Tk()
root.title("Seleccionar Modo de Ejecución")

tk.Label(root, text="Seleccione el modo de ejecución:").pack(pady=10)

btn_real = tk.Button(root, text="Modo Real", command=lambda: select_mode("1"))
btn_real.pack(pady=5)

btn_backtest = tk.Button(root, text="Modo Backtest", command=lambda: select_mode("2"))
btn_backtest.pack(pady=5)

root.mainloop()

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
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "USDMXN", "GDAXI"]  # Lista de activos a operar
models = {}

# Verificar y cargar modelos existentes
def load_models():
    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            models[symbol] = load_model(model_path)
            print(f"Modelo para {symbol} cargado correctamente.")
        else:
            print(f"Modelo no encontrado para {symbol}. Será generado en modo Backtest si es necesario.")

def generate_model(symbol, df):
    print(f"Generando modelo para {symbol}...")
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(10, len(df)):
        X.append(df.iloc[i-10:i][features].values)
        y.append(df.iloc[i]['close'])
    
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 7)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    model.save(f"lstm_model_{symbol}.keras")
    models[symbol] = model
    print(f"Modelo para {symbol} generado y guardado correctamente.")

def stop_trading():
    global running
    running = False
    print("Se ha detenido la ejecución del programa.")

def get_market_data(symbol, n_candles=200, save_to_csv=False):
    print(f"Obteniendo datos de {symbol}...")
    if not mt5.initialize():
        print("Error al inicializar MT5")
        mt5.shutdown()
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_candles)
    if rates is None:
        print(f"Error al obtener datos del mercado para {symbol}.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    if save_to_csv:
        df.to_csv(f"market_data_{symbol}.csv", index=False)
        print(f"Datos guardados en market_data_{symbol}.csv")
    return df

def add_technical_indicators(df):
    print("Calculando indicadores técnicos...")
    df['returns'] = df['close'].pct_change()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['returns'].rolling(14).mean() / df['returns'].rolling(14).std()))
    df['volatility'] = df['returns'].rolling(window=10).std()
    df.dropna(inplace=True)
    print(f"Indicadores técnicos calculados correctamente. Dataframe resultante: {df.shape}")
    return df

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
            predicted_price = models[symbol].predict(X)[0][0]
            print(f"Predicción para {symbol}: {predicted_price}")
        time.sleep(5)

def backtest():
    print("Iniciando Backtest...")
    for symbol in SYMBOLS:
        df = get_market_data(symbol, n_candles=2000, save_to_csv=True)
        if df is None:
            continue
        df = add_technical_indicators(df)
        if symbol not in models:
            generate_model(symbol, df)
    print("Backtest finalizado. Ahora puedes ejecutar en modo Real.")

# Cargar modelos al iniciar
load_models()

if mode == "1":
    real_trading()
elif mode == "2":
    backtest()
