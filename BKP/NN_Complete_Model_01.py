import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

running = True
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]  # Lista de activos a operar

def stop_trading():
    global running
    running = False

def get_market_data(symbol, n_candles=200):
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
    print(f"Datos obtenidos correctamente para {symbol}.")
    return df

def send_order(symbol, action):
    print(f"Enviando orden {action} para {symbol}...")
    lot_size = 0.1
    price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
    sl = price - 0.001 if action == "BUY" else price + 0.001
    tp = price + 0.002 if action == "BUY" else price - 0.002
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": "AI Trading",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Error al enviar orden: {result.comment}")
    else:
        print(f"Orden {action} ejecutada con éxito para {symbol}")

def select_mode():
    def set_mode_real():
        global mode
        mode = "1"
        root.destroy()
    
    def set_mode_backtest():
        global mode
        mode = "2"
        root.destroy()
    
    root = tk.Tk()
    root.title("Seleccionar Modo de Ejecución")
    tk.Label(root, text="Seleccione el modo de ejecución", font=("Arial", 12)).pack(pady=10)
    tk.Button(root, text="Real", command=set_mode_real, font=("Arial", 12), bg="green", fg="white").pack(pady=5)
    tk.Button(root, text="Backtest", command=set_mode_backtest, font=("Arial", 12), bg="blue", fg="white").pack(pady=5)
    tk.Button(root, text="DETENER", command=stop_trading, font=("Arial", 12), bg="red", fg="white").pack(pady=20)
    root.mainloop()

# Mostrar selección de modo antes de iniciar cualquier proceso
select_mode()

models = {}
for symbol in SYMBOLS:
    model_file = f"lstm_model_{symbol}.h5"
    if not os.path.exists(model_file):
        print(f"Error: No se encontró el modelo {model_file} para {symbol}. Asegúrese de entrenarlo antes de ejecutarlo.")
        continue
    print(f"Cargando modelo {model_file}...")
    models[symbol] = load_model(model_file, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    print(f"Modelo para {symbol} cargado correctamente.")

if mode == "1":  # Modo Real
    print("Ejecutando en modo real...")
    scaler = MinMaxScaler()
    while running:
        for symbol in SYMBOLS:
            print(f"Solicitando datos del mercado para {symbol}...")
            df = get_market_data(symbol, n_candles=50)
            if df is None or df.empty:
                print(f"No se pudieron obtener datos del mercado para {symbol}. Reintentando en la siguiente iteración...")
                continue
            
            print(f"Procesando datos para {symbol}...")
            df['returns'] = df['close'].pct_change()
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['RSI'] = 100 - (100 / (1 + df['returns'].rolling(14).mean() / df['returns'].rolling(14).std()))
            df['volatility'] = df['returns'].rolling(window=10).std()
            df.dropna(inplace=True)
            
            # Normalizar datos
            df_scaled = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'tick_volume', 'returns', 'SMA_10', 'SMA_50', 'RSI', 'volatility']])
            input_data = np.array([df_scaled[-10:, :7]])  # Últimos 10 pasos de tiempo con las primeras 7 características
            
            print(f"Realizando predicción para {symbol}...")
            predicted_price = models[symbol].predict(input_data)[0][0]
            predicted_price = predicted_price * (df['close'].max() - df['close'].min()) + df['close'].min()
            actual_price = df.iloc[-1]['close']
            
            print(f"Predicted Price: {predicted_price}, Actual Price: {actual_price} for {symbol}")
            
            with open(f"predictions_{symbol}.csv", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{symbol},{actual_price},{predicted_price}\n")
            
            if predicted_price > actual_price:
                print(f"Enviando orden de COMPRA para {symbol}")
                send_order(symbol, "BUY")
            elif predicted_price < actual_price:
                print(f"Enviando orden de VENTA para {symbol}")
                send_order(symbol, "SELL")
        
        print("Esperando 30 segundos antes de la siguiente iteración...")
        time.sleep(30)
