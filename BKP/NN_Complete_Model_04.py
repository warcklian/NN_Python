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
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "USDMXN", "GDAXI"]
models = {}
open_orders = {}

# Cargar modelos

def load_models():
    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            models[symbol] = load_model(model_path)
            print(f"Modelo para {symbol} cargado correctamente.")
        else:
            print(f"Modelo no encontrado para {symbol}. Será generado en modo Backtest si es necesario.")

# Guardar predicciones en CSV por símbolo

def save_predictions(symbol, prediction, actual_price):
    filename = f"predictions_realtime_{symbol}.csv"
    df = pd.DataFrame([[symbol, prediction, actual_price, time.time()]], columns=["symbol", "prediction", "actual_price", "timestamp"])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# Obtener datos del mercado

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

# Enviar órdenes

def send_order(symbol, action, predicted_change):
    print(f"Intentando enviar orden {action} en {symbol}...")
    if symbol in open_orders:
        print(f"Ya hay una orden activa en {symbol}. No se enviará otra hasta que la actual se cierre.")
        return
    
    order_type = mt5.ORDER_BUY if action == "BUY" else mt5.ORDER_SELL
    price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
    tp = price + predicted_change if action == "BUY" else price - predicted_change
    sl = price - (predicted_change * 0.5) if action == "BUY" else price + (predicted_change * 0.5)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.1,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": "AI Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        open_orders[symbol] = result.order
        print(f"Orden {action} ejecutada en {symbol}.")
    else:
        print(f"Error al enviar orden en {symbol}: {result.comment}")

# Verificar órdenes

def check_orders():
    for symbol, order_id in list(open_orders.items()):
        order_info = mt5.positions_get(ticket=order_id)
        if order_info:
            profit = order_info[0].profit
            if profit > 10:
                mt5.Close(symbol)
                del open_orders[symbol]
                print(f"Orden en {symbol} cerrada con ganancia de {profit}.")

# Modo Backtest

def backtest():
    print("Ejecutando Backtest y entrenando modelos...")
    for symbol in SYMBOLS:
        df = get_market_data(symbol, n_candles=1000, save_to_csv=True)
        if df is not None:
            features = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[features])
            X, y = [], []
            for i in range(len(scaled_data) - 10):
                X.append(scaled_data[i:i+10])
                y.append(scaled_data[i+10][3])
            X, y = np.array(X), np.array(y)
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(10, len(features))),
                LSTM(50),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=20, batch_size=32)
            model.save(f"lstm_model_{symbol}.keras")
            print(f"Modelo de {symbol} entrenado y guardado correctamente.")

# Modo Real

def real_trading():
    print("Iniciando ejecución en modo real...")
    while running:
        for symbol in SYMBOLS:
            df = get_market_data(symbol, n_candles=200)
            if df is None:
                continue
            predicted_price = models[symbol].predict(np.array([df.iloc[-10:].values]))[0][0]
            predicted_change = abs(predicted_price - df.iloc[-1]['close'])
            save_predictions(symbol, predicted_price, df.iloc[-1]['close'])
            send_order(symbol, "BUY" if predicted_price > df.iloc[-1]['close'] else "SELL", predicted_change)
            check_orders()
        time.sleep(5)

load_models()
if mode == "1":
    real_trading()
elif mode == "2":
    backtest()
