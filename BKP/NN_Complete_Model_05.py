import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Selección del modo de ejecución
def select_mode(mode_choice):
    global mode
    mode = mode_choice
    root.destroy()

root = tk.Tk()
root.title("Seleccionar Modo de Ejecución")

tk.Label(root, text="Seleccione el modo de ejecución:").pack(pady=10)

btn_real = tk.Button(root, text="Modo Real", command=lambda: select_mode("1"))
btn_real.pack(pady=5)

btn_backtest = tk.Button(root, text="Modo Backtest", command=lambda: select_mode("2"))
btn_backtest.pack(pady=5)

root.mainloop()

# Configuración de TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detectadas: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No se detectaron GPUs. Usando CPU.")

running = True
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "USDMXN", "GDAXI"]
models = {}
open_orders = {}

# Cargar modelos entrenados
def load_models():
    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            models[symbol] = load_model(model_path)
            print(f"Modelo para {symbol} cargado correctamente.")
        else:
            print(f"Modelo no encontrado para {symbol}. Será generado en modo Backtest si es necesario.")

# Obtener datos de mercado
def get_market_data(symbol, n_candles=200):
    if not mt5.initialize():
        print("Error al inicializar MT5")
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_candles)
    if rates is None:
        print(f"Error al obtener datos del mercado para {symbol}.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calcular tamaño de lote dinámico
def calculate_lot_size(symbol, risk=1):
    account_info = mt5.account_info()
    balance = account_info.balance
    risk_amount = balance * (risk / 100)
    point = mt5.symbol_info(symbol).point
    lot_size = risk_amount / (point * 100000)  # Ajustado para Forex
    return round(min(max(lot_size, 0.01), 1), 2)  # Límite de 0.01 a 1 lote

# Enviar orden de compra o venta
def send_order(symbol, action, predicted_change):
    print(f"Intentando enviar orden {action} en {symbol}...")

    existing_orders = mt5.positions_get(symbol=symbol)
    if existing_orders:
        for order in existing_orders:
            if (order.type == 0 and action == "BUY") or (order.type == 1 and action == "SELL"):
                print(f"Ya hay una orden {action} activa en {symbol}. No se enviará otra.")
                return
    
    if not mt5.initialize():
        print("Error al conectar con MT5")
        return
    
    order_type = 0 if action == "BUY" else 1
    tick_info = mt5.symbol_info_tick(symbol)
    if not tick_info:
        print(f"Error al obtener precio para {symbol}")
        return
    
    price = tick_info.ask if action == "BUY" else tick_info.bid
    lot_size = 0.10

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 123456,
        "comment": "AI Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    
    print(f"Enviando orden: {request}")
    
    result = mt5.order_send(request)
    
    if result is None:
        print(f"Error: mt5.order_send() devolvió None para {symbol}.")
        print(f"Detalles de la solicitud: {request}")
        return
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Orden {action} ejecutada en {symbol}. Precio: {price}, Volumen: {lot_size}")
    else:
        print(f"Error al enviar orden en {symbol}: {result.comment} (Código: {result.retcode})")

# Ejecutar Trading en tiempo real
def real_trading():
    while running:
        for symbol in SYMBOLS:
            df = get_market_data(symbol, n_candles=200)
            if df is None:
                continue
            df_numeric = df.iloc[-10:].select_dtypes(include=[np.number])
            predicted_price = models[symbol].predict(np.array([df_numeric.values.astype(float)]))[0][0]
            send_order(symbol, "BUY" if predicted_price > df.iloc[-1]['close'] else "SELL", abs(predicted_price - df.iloc[-1]['close']))
        time.sleep(5)

# Backtesting
def backtest():
    print("Iniciando backtest...")
    for symbol in SYMBOLS:
        df = get_market_data(symbol, n_candles=1000)
        if df is None:
            continue
        df_numeric = df.select_dtypes(include=[np.number])
        predictions = models[symbol].predict(np.array([df_numeric.values.astype(float)]))
        df['predicted'] = predictions.flatten()
        df.to_csv(f"backtest_results_{symbol}.csv", index=False)
        print(f"Backtest completado para {symbol}. Resultados guardados.")

# Cargar modelos y ejecutar en el modo seleccionado
load_models()
if mode == "1":
    real_trading()
elif mode == "2":
    backtest()
