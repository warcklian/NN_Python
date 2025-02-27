import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Obtener datos del mercado y guardarlos en CSV
def get_market_data(symbol, n_candles=1000):
    print(f"Obteniendo datos de {symbol} para Backtest...")
    if not mt5.initialize():
        print("Error al inicializar MT5")
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_candles)
    if rates is None:
        print(f"Error al obtener datos del mercado para {symbol}.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.to_csv(f"market_data_{symbol}.csv", index=False)
    return df

# Definir y entrenar modelo LSTM
def train_lstm_model(symbol):
    df = pd.read_csv(f"market_data_{symbol}.csv")
    if df is None or df.empty:
        print(f"No se puede entrenar el modelo para {symbol}, datos no encontrados.")
        return None
    
    df_numeric = df[['open', 'high', 'low', 'close', 'tick_volume']]
    X = []
    y = []
    lookback = 10
    
    for i in range(lookback, len(df_numeric) - 1):
        X.append(df_numeric.iloc[i - lookback:i].values)
        y.append(df_numeric.iloc[i + 1]['close'])
    
    X = np.array(X)
    y = np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    model.save(f"lstm_model_{symbol}.keras")
    print(f"Modelo para {symbol} entrenado y guardado.")
    return model

# Cargar modelos entrenados
def load_models():
    models = {}
    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            models[symbol] = load_model(model_path)
            print(f"Modelo para {symbol} cargado correctamente.")
        else:
            print(f"Modelo no encontrado para {symbol}. Será generado en modo Backtest.")
            models[symbol] = train_lstm_model(symbol)
    return models

# Ejecutar backtest
def backtest():
    print("Iniciando backtest...")
    for symbol in SYMBOLS:
        df = get_market_data(symbol, n_candles=1000)
        if df is None:
            continue
    
    models = load_models()
    for symbol in SYMBOLS:
        df = pd.read_csv(f"market_data_{symbol}.csv")
        df_numeric = df[['open', 'high', 'low', 'close', 'tick_volume']]
        X_test = []
        lookback = 10
        
        for i in range(lookback, len(df_numeric) - 1):
            X_test.append(df_numeric.iloc[i - lookback:i].values)
        
        X_test = np.array(X_test)
        
        predictions = models[symbol].predict(X_test)
        df.loc[lookback:lookback + len(predictions) - 1, 'predicted'] = predictions.flatten()
        df.to_csv(f"backtest_results_{symbol}.csv", index=False)
        print(f"Backtest completado para {symbol}. Resultados guardados.")

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "USDMXN"]
backtest()
