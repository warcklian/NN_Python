import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
#import tkinter as tk
import tkinter as ttk
from tkinter import messagebox
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Input
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from tqdm import tqdm
import seaborn as sns
import requests
from datetime import datetime
import csv
import matplotlib
matplotlib.use('Agg')  # Evita problemas en hilos secundarios
import matplotlib.pyplot as plt
from keras_tuner import RandomSearch  # Solo lo mantenemos si realmente se usa
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
from sklearn.preprocessing import StandardScaler  # Asegurar que está importado
import logging
import joblib





magicNumber = 789654
my_candles = 5000

# Configuración de los parámetros de trading
SL_PIPS = 500  # Stop Loss en pips
TP_PIPS = 1000  # Take Profit en pips
TRAILING_STOP_PIPS = 500  # Distancia mínima del SL respecto al precio actual
TRAILING_STEP_PIPS = 200  # Pips que deben moverse antes de actualizar el SL

# Configuración de modelo LSTM optimizado
LSTM_UNITS_1 = 128  # Aumentamos a 128 neuronas
LSTM_UNITS_2 = 64  # Segunda capa con 64 neuronas
USE_BIDIRECTIONAL = True  # Activar LSTM bidireccional
DROPOUT_RATE_1 = 0.1  # Reducimos dropout
DROPOUT_RATE_2 = 0.1  # Reducimos dropout
INPUT_TIMESTEPS = 10  # Mantener secuencia de 10 timesteps
INPUT_FEATURES = 9  # Mantener 9 features
DENSE_UNITS = 1  # Capa de salida con 1 neurona

# Hiperparámetros de entrenamiento optimizados
EPOCHS = 50  # Aumentamos las épocas
BATCH_SIZE = 32  # Reducimos batch size para mayor precisión

# Variables globales
models = {}  # Diccionario para almacenar modelos de ML
SYMBOLS = ["EURUSD", "USDJPY", "USDMXN", "XAUUSD", "XAGUSD"]
running = False  # Control de ejecución de simulación

# Crear una ventana para la selección del modo

# Configurar logging
log_filename = "trading_bot.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_message(message, level="info"):
    """Registra mensajes en el log y los muestra en la consola."""
    print(message)  # Mostrar en consola

    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)


print("Buscando GPUs disponibles...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detectadas: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error al configurar memoria de GPU: {e}")
    strategy = tf.distribute.MirroredStrategy()
else:
    print("No se detectaron GPUs. Usando CPU.")
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")

@tf.function(reduce_retracing=True)
def predict_price(model, X_input):
    """Función optimizada para reducir el retracing de TensorFlow."""
    return model(X_input, training=False)



running = False

# Definir la función de pérdida antes de cargar modelos
def custom_loss_function(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

tf.keras.utils.get_custom_objects()["custom_loss_function"] = custom_loss_function


retraining_status = {}  # Diccionario para controlar el estado del reentrenamiento

def retrain_model(symbol):
    """Reentrena el modelo LSTM para un símbolo específico integrando datos históricos desde CSV y nuevos datos del mercado."""
    global retraining_status

    if retraining_status.get(symbol, False):
        print(f"El modelo para {symbol} ya se está reentrenando. Esperando a que finalice...")
        return  
    
    retraining_status[symbol] = True  
    print(f"Reentrenando modelo para {symbol}...")

    # Nombre del archivo histórico
    historical_data_filename = f"historical_data_{symbol}.csv"

    # Intentar cargar datos históricos si el archivo existe
    if os.path.exists(historical_data_filename):
        df_historical = pd.read_csv(historical_data_filename)
    else:
        df_historical = pd.DataFrame()  # Crear un DataFrame vacío si no hay historial

    # Obtener datos recientes del mercado
    df_recent = get_market_data(symbol, my_candles)

    if df_recent is None or df_recent.empty:
        print(f"No se pudo obtener datos recientes para reentrenar {symbol}.")
        retraining_status[symbol] = False
        return

    # Integrar indicadores técnicos a los datos recientes
    df_recent = add_technical_indicators(df_recent)

    # Concatenar datos históricos con datos recientes
    df = pd.concat([df_historical, df_recent], ignore_index=True)

    # Eliminar duplicados basándose en la columna de tiempo si existe
    if "time" in df.columns:
        df.drop_duplicates(subset=["time"], keep="last", inplace=True)

    # Lista de características a utilizar
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']

    # Normalización de los datos
    scaler_X = MinMaxScaler()
    df[features] = scaler_X.fit_transform(df[features])

    scaler_y = MinMaxScaler()
    df['close_scaled'] = scaler_y.fit_transform(df[['close']])

    # Preparación de datos para entrenar
    X, y = [], []
    for i in range(INPUT_TIMESTEPS, len(df)):
        X.append(df.iloc[i-INPUT_TIMESTEPS:i][features].values)
        y.append(df.iloc[i]['close'])

    X, y = np.array(X), np.array(y)

    # Construcción del modelo LSTM
    model = keras.models.Sequential()

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True), input_shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))
    else:
        model.add(LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))

    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_RATE_1))

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False)))
    else:
        model.add(LSTM(LSTM_UNITS_2, return_sequences=False))

    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_RATE_2))

    model.add(Dense(DENSE_UNITS))

    model.compile(optimizer='adam', loss='mse')

    # Entrenar modelo con los datos combinados
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Validar modelo antes de guardarlo
    print(f"Evaluando rendimiento del modelo reentrenado para {symbol}...")
    X_test, y_true = [], []
    for i in range(10, len(df)):
        X_test.append(df.iloc[i-10:i][features].values)
        y_true.append(df.iloc[i]['close'])

    X_test, y_true = np.array(X_test), np.array(y_true)
    y_pred = model.predict(X_test).flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Resultados del modelo reentrenado para {symbol}:")
    print(f"   - MSE: {mse:.5f}")
    print(f"   - MAE: {mae:.5f}")
    print(f"   - R²: {r2:.5f}")

    # Si el modelo tiene un R² negativo, no se reemplaza el anterior
    if r2 < 0:
        print(f"Advertencia: El modelo reentrenado para {symbol} tiene R² negativo ({r2:.5f}). No será reemplazado.")
        retraining_status[symbol] = False
        return

    # Guardar modelo y actualizar datos históricos
    model.save(f"lstm_model_{symbol}.keras")
    models[symbol] = model
    df.to_csv(historical_data_filename, index=False)

    print(f"Modelo reentrenado y datos históricos actualizados correctamente para {symbol}.")
    retraining_status[symbol] = False





# Verificar y cargar modelos existentes
def load_models():
    """Carga modelos LSTM y los reentrena si hay errores al cargarlos."""
    print("Cargando modelos...")

    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            try:
                models[symbol] = load_model(model_path, custom_objects={"custom_loss_function": custom_loss_function})
                print(f"Modelo para {symbol} cargado correctamente.")
            except Exception as e:
                print(f"Error al cargar el modelo para {symbol}: {e}. Reentrenando...")
                retrain_model(symbol)
        else:
            print(f"Modelo no encontrado para {symbol}. Generando uno nuevo...")
            retrain_model(symbol)

    print(f"Modelos cargados en memoria: {models.keys()}")




def add_technical_indicators(df):
    """Calcula indicadores técnicos y maneja errores de datos."""
    print("Calculando indicadores técnicos...")

    if df is None or df.empty:
        print("Error: No se puede calcular indicadores en un DataFrame vacío.")
        return df

    try:
        df['returns'] = df['close'].pct_change()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = 100 - (100 / (1 + df['returns'].rolling(14).mean() / df['returns'].rolling(14).std()))
        df['RSI'] = df['RSI'].fillna(50)  # Rellenar valores NaN
        df['volatility'] = df['returns'].rolling(window=10).std()
        df.dropna(inplace=True)

        print(f"Indicadores técnicos calculados correctamente. DataFrame resultante: {df.shape}")
    except Exception as e:
        print(f"Error al calcular indicadores técnicos: {e}")

    return df



def optimize_model(symbol, df):
    """Optimiza los hiperparámetros del modelo LSTM para mejorar su rendimiento."""
    print(f"Optimizando modelo para {symbol}...")

    features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(10, len(df)):
        X.append(df.iloc[i-10:i][features].values)
        y.append(df.iloc[i]['close'])

    X, y = np.array(X), np.array(y)

    def build_model(hp):
        model = keras.models.Sequential() #model = Sequential()        
        model.add(Input(shape=(10, 9)))  # Define la forma de entrada explícitamente
        model.add(LSTM(hp.Int('units', min_value=32, max_value=128, step=32), return_sequences=True))
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(LSTM(hp.Int('units', min_value=32, max_value=128, step=32), return_sequences=False))
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='model_tuning',
        project_name=f"opt_{symbol}"
    )

    tuner.search(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Mejores hiperparámetros para {symbol}: {best_hps.values}")

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    best_model.save(f"lstm_model_{symbol}.keras")

    models[symbol] = best_model
    print(f"Modelo optimizado para {symbol} guardado correctamente.")


def generate_model(symbol, df):
    """Genera un modelo LSTM o ejecuta la optimización de hiperparámetros si es necesario."""
    print(f"Generando modelo para {symbol}...")

    # Asegurar que el DataFrame tenga los indicadores calculados
    df = add_technical_indicators(df)

    model_path = f"lstm_model_{symbol}.keras"
    if not os.path.exists(model_path):
        print(f"No se encontró modelo para {symbol}. Iniciando optimización...")
        optimize_model(symbol, df)
    else:
        print(f"Modelo para {symbol} ya existe. Cargando...")
        models[symbol] = load_model(model_path)


def check_mt5_connection():
    """Verifica la conexión con MetaTrader 5 y la reinicia si es necesario."""
    if not mt5.initialize():
        error_message = " Error al inicializar MT5. Intentando reiniciar..."
        print(error_message)
        log_error(error_message)
        mt5.shutdown()
        time.sleep(2)  # Espera 2 segundos antes de reintentar
        if not mt5.initialize():
            error_message = " Falló la reconexión a MT5. Verifica tu terminal."
            print(error_message)
            log_error(error_message)
            return False
    return True

def get_market_data_from_mt5(symbol, n_candles):
    """Obtiene datos en vivo desde MT5 y maneja errores."""
    print(f"Obteniendo datos en vivo para {symbol} desde MT5...")

    if not check_mt5_connection():
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_candles)
    
    if rates is None or len(rates) == 0:
        print(f"Error al obtener datos del mercado para {symbol}.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    if df.isnull().values.any():
        print("Advertencia: Se encontraron valores NaN. Eliminando registros problemáticos...")
        df.dropna(inplace=True)  # En lugar de llamar a otra función que podría generar recursión

    return df

def evaluate_model(symbol, df):
    """Evalúa el modelo en datos recientes y usa predicciones previas si existen."""
    
    if df is None or df.empty:
        print(f"Advertencia: No hay datos suficientes para evaluar {symbol}.")
        return

    print(f"\nColumnas disponibles en df antes de escalar ({symbol}):", df.columns)

    required_features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']

    # Cargar predicciones previas si existen
    predictions_filename = f"predictions_{symbol}.csv"

    if os.path.exists(predictions_filename):
        df_predictions = pd.read_csv(predictions_filename)
        df_predictions['Fecha'] = pd.to_datetime(df_predictions['Fecha'])
        df_predictions = df_predictions[df_predictions["Símbolo"] == symbol]

        # Integrar las predicciones previas con el dataset actual
        df = pd.merge(df, df_predictions, left_on="time", right_on="Fecha", how="left")

        if 'Predicción' in df.columns:
            required_features.append('Predicción')

    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"Error: Las siguientes columnas faltan en el dataset para {symbol}: {missing_features}")
        return

    try:
        scaler_X = StandardScaler()
        df_scaled = df.copy()
        df_scaled[required_features] = scaler_X.fit_transform(df_scaled[required_features])

        # Preparar datos para evaluación
        X_test, y_true = [], []
        for i in range(10, len(df_scaled)):
            X_test.append(df_scaled.iloc[i-10:i][required_features].values)
            y_true.append(df_scaled.iloc[i]['close'])

        X_test, y_true = np.array(X_test), np.array(y_true)

        # Realizar predicciones
        y_pred = models[symbol].predict(X_test).flatten()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"Resultados de evaluación para {symbol}:")
        print(f"   - MSE: {mse:.5f}")
        print(f"   - MAE: {mae:.5f}")
        print(f"   - R²: {r2:.5f}")

        # Guardar predicción
        save_prediction(symbol, y_pred[-1], y_true[-1])

        if r2 < 0:
            print(f"Advertencia: R² negativo ({r2:.5f}) en {symbol}. Se recomienda reentrenar el modelo.")
            retrain_model(symbol)
        else:
            print(f"Evaluación completada para {symbol}. Modelo válido con R² de {r2:.5f}.")

    except Exception as e:
        print(f"Error al evaluar el modelo para {symbol}: {e}")













cached_market_data = {}

def get_cached_market_data(symbol, n_candles, cache_time=30):
    """Obtiene datos del mercado con caché, verificando errores en la fuente."""
    current_time = time.time()

    if symbol in cached_market_data:
        last_update, cached_df = market_data_cache[symbol]
        if current_time - last_update < cache_time:
            print(f" Usando datos en caché para {symbol}")
            return cached_df

    df = get_market_data_from_mt5(symbol, n_candles)
    
    if df is None or df.empty:
        print(f" No se pudieron obtener datos de {symbol}.")
        return None

    cached_market_data[symbol] = (current_time, df)
    return df


# Obtener datos de mercado con indicadores técnicos
market_data_cache = {}

def get_market_data(symbol, n_candles, save_to_csv=True):
    """Obtiene datos del mercado combinando datos en vivo, históricos y predicciones previas."""

    market_filename = f"market_data_{symbol}.csv"

    # Cargar datos históricos si existen
    if os.path.exists(market_filename):
        print(f"Cargando datos históricos desde {market_filename} para {symbol}...")
        df = pd.read_csv(market_filename)
        df['time'] = pd.to_datetime(df['time'])
    else:
        df = pd.DataFrame()  # Crear DataFrame vacío si no hay historial

    # Obtener datos en vivo desde MT5
    df_live = get_market_data_from_mt5(symbol, n_candles)
    
    if df_live is None or df_live.empty:
        print(f"No se pudo obtener datos en vivo de {symbol}.")
        if df.empty:
            print(f"No hay datos disponibles en {market_filename}.")
            return None
        return df  # Usar solo datos históricos si no hay datos en vivo

    # Combinar datos históricos con los nuevos y eliminar duplicados
    df = pd.concat([df, df_live]).drop_duplicates(subset=['time']).reset_index(drop=True)

    # Aplicar indicadores técnicos
    df = add_technical_indicators(df)

    # Guardar en CSV si es necesario
    if save_to_csv:
        try:
            temp_filename = f"{market_filename}.temp"
            df.to_csv(temp_filename, index=False)  # Guardar en archivo temporal
            os.replace(temp_filename, market_filename)  # Reemplazar archivo original
            print(f"Datos de {symbol} actualizados y guardados en {market_filename}")
        except Exception as e:
            print(f"Error al guardar el archivo CSV {market_filename}: {e}")

    return df








def backtest():
    """Ejecuta un backtest con validación de datos y carga modelos solo si no están cargados."""
    global models
    if not models:  # Solo carga los modelos si aún no están cargados
        print("Cargando modelos antes del Backtest...")
        load_models()

    print("Iniciando Backtest...")
    for symbol in SYMBOLS:
        df = get_market_data(symbol, my_candles, save_to_csv=True)
        
        if df is None or df.empty:
            print(f"No se pudieron obtener datos para {symbol}. Saltando...")
            continue
        
        df = add_technical_indicators(df)
        print(f"Evaluando modelo para {symbol}...")
        evaluate_model(symbol, df)
    
    print("Backtest finalizado.")





def initialize_models():
    """Inicializa los modelos y guarda los datos en CSV."""
    for symbol in SYMBOLS:
        df = get_market_data(symbol, my_candles)

        if df is None or df.empty:
            print(f"No se pudo obtener datos para {symbol}. Saltando generación de modelo.")
            continue
        
        df = add_technical_indicators(df)

        # Guardar datos en CSV para análisis
        csv_filename = f"market_data_{symbol}.csv"
        df.to_csv(market_filename, index=False, mode='w', encoding='utf-8', errors='replace')
        print(f"Datos de {symbol} guardados en {csv_filename}.")

        # Verificar si el modelo ya existe
        model_path = f"lstm_model_{symbol}.keras"
        if not os.path.exists(model_path):
            print(f"No se encontró modelo para {symbol}. Generando uno nuevo...")
            generate_model(symbol, df)
        else:
            print(f"Modelo para {symbol} ya existe. Evaluando...")
            evaluate_model(symbol, df)

    print("Modelos inicializados y datos guardados en CSV.")




def send_order(symbol, action, lot_size=0.1):
    """Envía una orden de compra o venta en MT5 con detalles completos de SL, TP y trailing stop, registrando logs."""
    log_message(f"\n##### Enviando orden {action} en {symbol} #####")

    if not check_mt5_connection():
        log_message("No se puede enviar la orden. MT5 no está conectado.", "error")
        return

    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    tick_info = mt5.symbol_info_tick(symbol)

    if tick_info is None or tick_info.ask is None or tick_info.bid is None:
        log_message(f"Error al obtener precio para {symbol}", "error")
        return

    price = tick_info.ask if action == "BUY" else tick_info.bid
    point = mt5.symbol_info(symbol).point

    if price is None or point is None:
        log_message(f"Error: No se pudo obtener el precio o el punto para {symbol}.", "error")
        return

    sl = price - SL_PIPS * point if action == "BUY" else price + SL_PIPS * point
    tp = price + TP_PIPS * point if action == "BUY" else price - TP_PIPS * point

    log_message(f"Precio de ejecución: {price:.5f}")
    log_message(f"Stop Loss (SL): {sl:.5f}")
    log_message(f"Take Profit (TP): {tp:.5f}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": magicNumber,
        "comment": "Deep Market Predictor AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if result is None:
        log_message(f"Error al enviar orden en {symbol}. MT5 podría no estar conectado.", "error")
        return

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log_message(f"Orden {action} ejecutada en {symbol}. Precio: {price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
    else:
        log_message(f"Error al enviar orden en {symbol}: {result.comment}", "error")






def check_existing_order(symbol):
    """Verifica si ya hay una orden abierta en el símbolo, pero solo si fue enviada por el bot."""
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            if pos.symbol == symbol and pos.magic == magicNumber:  # Solo contar órdenes con el magic number del bot
                print(f"Ya hay una orden abierta en {symbol} enviada por el bot. No se enviará otra.")
                return True
    return False


# Verificar y cargar modelos existentes
def load_models():
    """Carga modelos LSTM y los reentrena si hay errores al cargarlos."""
    print("Cargando modelos...")

    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            try:
                models[symbol] = load_model(model_path, custom_objects={"custom_loss_function": custom_loss_function})
                print(f"Modelo para {symbol} cargado correctamente.")
            except Exception as e:
                print(f"Error al cargar el modelo para {symbol}: {e}. Reentrenando...")
                retrain_model(symbol)
        else:
            print(f"Modelo no encontrado para {symbol}. Generando uno nuevo...")
            retrain_model(symbol)

    print(f"Modelos cargados en memoria: {models.keys()}")



def get_market_data_from_mt5(symbol, n_candles):
    """Obtiene datos en vivo desde MT5 y maneja errores."""
    print(f" Obteniendo datos en vivo para {symbol} desde MT5...")

    if not check_mt5_connection():
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_candles)
    
    if rates is None or len(rates) == 0:
        print(f" Error al obtener datos del mercado para {symbol}.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    if df.isnull().values.any():
        print(" Se encontraron valores NaN. Eliminando registros problemáticos...")
        df.dropna(inplace=True)

    return df


def clean_market_data(df):
    """Limpia los datos del mercado, manejando valores NaN antes de usarlos en predicciones."""
    if df.isnull().values.any():
        print(" Se encontraron valores NaN en los datos. Se están limpiando...")
        df.fillna(method="ffill", inplace=True)  # Rellenar con valores previos
        df.fillna(method="bfill", inplace=True)  # Si hay valores NaN al inicio, rellenar con los siguientes
    return df


def save_prediction(symbol, predicted_price, actual_price):
    """Guarda las predicciones en un archivo CSV específico por símbolo."""
    predictions_filename = f"predictions_{symbol}.csv"

    error_abs = abs(predicted_price - actual_price)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    data = [[symbol, predicted_price, actual_price, error_abs, timestamp]]
    df = pd.DataFrame(data, columns=["Símbolo", "Predicción", "Precio Real", "Error Absoluto", "Fecha"])
    
    if os.path.exists(predictions_filename):
        df.to_csv(predictions_filename, mode='a', header=False, index=False)
    else:
        df.to_csv(predictions_filename, index=False)

    print(f"Predicción guardada en {predictions_filename}: {predicted_price:.5f} (Real: {actual_price:.5f})")






# Diccionario para almacenar predicciones en memoria
predictions_data = {}

def plot_predictions():
    """Genera una gráfica en tiempo real con las predicciones y los precios reales."""
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.set_title("Predicciones del Modelo vs Precios Reales")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Precio")

        for symbol, data in predictions_data.items():
            if len(data["timestamps"]) > 1:
                ax.plot(data["timestamps"], data["predictions"], label=f"Predicción {symbol}", linestyle="dashed")
                ax.plot(data["timestamps"], data["real_prices"], label=f"Real {symbol}")

        if any(len(data["timestamps"]) > 1 for data in predictions_data.values()):  # Solo agregar leyenda si hay datos
            ax.legend()

    
    try:
        plt.show()
    except UserWarning:
        print("Modo no interactivo detectado, guardando gráfica en 'predictions_plot.png'")
        plt.savefig("predictions_plot.png")


def close_order(position):
    """Cierra una orden abierta en MetaTrader 5, pero solo si pertenece al bot."""
    if position.magic != magicNumber:  # Verifica que la orden fue enviada por el bot
        print(f"Ignorando cierre de orden en {position.symbol} (no pertenece al bot).")
        return
    
    symbol = position.symbol
    ticket = position.ticket
    lot_size = position.volume
    order_type = position.type
    price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 10,
        "magic": magicNumber,  # Confirmamos que la orden pertenece al bot
        "comment": "Cierre Automático",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(close_request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f" Orden en {symbol} cerrada con éxito. Precio de cierre: {price}")
    else:
        print(f" Error al cerrar orden en {symbol}: {result.comment}")


def check_orders():
    """Revisa las órdenes abiertas y las cierra si alcanzan los niveles de TP o SL dinámicos,
    pero solo si fueron enviadas por el bot (según el magic number)."""
    
    if not check_mt5_connection():
        log_message("No se pudo revisar órdenes. MT5 no está conectado.", "error")
        return

    positions = mt5.positions_get()
    if not positions:
        log_message("No hay órdenes abiertas en este momento.")
        return

    for position in positions:
        if position.magic != magicNumber:  # Ignorar posiciones que no sean del bot
            continue

        symbol = position.symbol
        profit = position.profit

        log_message(f"Revisando orden abierta en {symbol}: Ganancia actual = {profit:.2f} USD")

        # Definir umbrales de cierre automático (configurables)
        take_profit_threshold = 15  # Cerrar si la ganancia supera 15 USD
        stop_loss_threshold = -10   # Cerrar si la pérdida es mayor a -10 USD

        # Cerrar posición si se cumplen las condiciones
        if profit >= take_profit_threshold:
            log_message(f"Cierre automático: {symbol} alcanzó Take Profit (+{profit:.2f} USD). Cerrando orden...")
            close_order(position)
        elif profit <= stop_loss_threshold:
            log_message(f"Cierre automático: {symbol} alcanzó Stop Loss ({profit:.2f} USD). Cerrando orden...")
            close_order(position)



def save_trading_statistics():
    """Guarda estadísticas del trading en un archivo CSV para análisis de rendimiento."""
    filename = "trading_statistics.csv"

    if not check_mt5_connection():
        return

    positions = mt5.history_deals_get(time.time() - 7 * 24 * 3600, time.time())  # Últimos 7 días
    if not positions:
        print("No se encontraron operaciones en el historial.")
        return

    total_trades = len(positions)
    profitable_trades = sum(1 for deal in positions if deal.profit > 0)
    losing_trades = total_trades - profitable_trades
    total_profit = sum(deal.profit for deal in positions)
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

    data = [[time.strftime('%Y-%m-%d %H:%M:%S'), total_trades, profitable_trades, losing_trades, total_profit, avg_profit, win_rate]]
    df = pd.DataFrame(data, columns=["Fecha", "Total Trades", "Ganadoras", "Perdedoras", "Ganancia Total", "Promedio Ganancia", "Tasa de Éxito (%)"])

    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(market_filename, index=False, mode='w', encoding='utf-8', errors='replace')

    print(f"Estadísticas de trading guardadas en {filename}.")

class PaperTrading:
    """Simulador de trading sin ejecutar órdenes reales."""

    def __init__(self):
        self.balance = 10000  # Capital inicial ficticio
        self.positions = {}  # Almacena órdenes abiertas

    def send_order(self, symbol, action, entry_price, lot_size=0.1):
        """Simula la ejecución de una orden de compra o venta."""
        self.positions[symbol] = {
            "action": action,
            "entry_price": entry_price,
            "lot_size": lot_size
        }
        print(f"✅ Orden simulada {action} en {symbol} ejecutada a {entry_price}")

    def check_orders(self):
        """Revisa y cierra órdenes simuladas si alcanzan niveles de TP o SL."""
        closed_positions = []

        for symbol, position in list(self.positions.items()):
            action = position["action"]
            entry_price = position["entry_price"]
            lot_size = position["lot_size"]

            # Obtener el precio actual
            tick_info = mt5.symbol_info_tick(symbol)
            if tick_info is None:
                print(f"⚠ No se pudo obtener el precio actual de {symbol}.")
                continue

            current_price = tick_info.bid if action == "BUY" else tick_info.ask
            profit = (current_price - entry_price) * 100 if action == "BUY" else (entry_price - current_price) * 100

            # Simulación de Take Profit y Stop Loss
            TP = 50  # Ganancia de 50 USD
            SL = -30  # Pérdida de 30 USD

            if profit >= TP:
                self.balance += profit
                print(f"🎯 {symbol} alcanzó Take Profit (+{profit:.2f} USD). Cerrando orden.")
                closed_positions.append(symbol)

            elif profit <= SL:
                self.balance += profit
                print(f"🛑 {symbol} alcanzó Stop Loss ({profit:.2f} USD). Cerrando orden.")
                closed_positions.append(symbol)

        # Eliminar órdenes cerradas
        for symbol in closed_positions:
            del self.positions[symbol]

    def show_balance(self):
        """Muestra el balance simulado actualizado."""
        print(f"💰 Balance simulado: {self.balance:.2f} USD")



#class PaperTrading:
    """Simula órdenes de compra y venta sin ejecutarlas en MetaTrader 5."""
    
    def __init__(self):
        self.balance = 10000  # Capital inicial ficticio
        self.positions = {}

    def send_order(self, symbol, action, lot_size=0.1):
        """Simula la ejecución de una orden de compra o venta."""
        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            print(f"Error al obtener datos de {symbol}.")
            return

        price = tick_info.ask if action == "BUY" else tick_info.bid
        self.positions[symbol] = {"action": action, "entry_price": price, "lot_size": lot_size}
        print(f"Simulación: Orden {action} en {symbol} ejecutada a {price}")

    def check_orders(self):
        """Revisa y cierra órdenes simuladas si alcanzan niveles de TP o SL."""
        for symbol, position in list(self.positions.items()):
            tick_info = mt5.symbol_info_tick(symbol)
            if tick_info is None:
                continue

            current_price = tick_info.bid
            entry_price = position["entry_price"]
            profit = (current_price - entry_price) * 100 if position["action"] == "BUY" else (entry_price - current_price) * 100

            if profit >= 50:
                self.balance += profit
                print(f"Simulación: {symbol} alcanzó TP con {profit} USD. Cerrando orden.")
                del self.positions[symbol]
            elif profit <= -30:
                self.balance += profit
                print(f"Simulación: {symbol} alcanzó SL con {profit} USD. Cerrando orden.")
                del self.positions[symbol]

    def show_balance(self):
        """Muestra el balance simulado actual."""
        print(f"Balance simulado: {self.balance} USD")


def simulation_thread():
    """Hilo de ejecución de la simulación."""
    global models

    # Cargar modelos si aún no están en memoria
    if not models:
        print("Cargando modelos antes de la Simulación...")
        load_models()

    paper_trading = PaperTrading()  # Instancia del simulador de trading
    print("Simulación iniciada...")

    while running:
        for symbol in SYMBOLS:
            print(f"\nProcesando {symbol} en la simulación...")

            df = get_market_data(symbol, my_candles)
            if df is None or df.empty:
                print(f"Advertencia: No hay datos suficientes para {symbol}. Omitiendo...")
                continue

            df = add_technical_indicators(df)
            required_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Error: Faltan columnas en {symbol}: {missing_columns}. Saltando...")
                continue

            trend = detect_trend(df)
            print(f"Tendencia detectada en {symbol}: {trend}")

            if trend == "LATERAL":
                print(f"Mercado lateral detectado en {symbol}. Omitiendo...")
                continue

            try:
                X_input = np.array([df.iloc[-10:][required_columns].values])
                predicted_price = models[symbol].predict(X_input)[0][0]
                actual_price = df.iloc[-1]['close']

                print(f"Predicción para {symbol}: {predicted_price}, Precio actual: {actual_price}")

                if validate_trade_signal(symbol, predicted_price, actual_price, df):
                    order_type = "BUY" if predicted_price > actual_price else "SELL"
                    print(f"📌 Ejecutando orden simulada: {order_type} en {symbol} a {actual_price}...")
                    paper_trading.send_order(symbol, order_type, actual_price)

            except Exception as e:
                print(f"Error en la simulación de {symbol}: {e}")
                continue

        # Revisar y cerrar órdenes abiertas
        paper_trading.check_orders()

        # Mostrar balance actualizado después de evaluar todas las operaciones
        paper_trading.show_balance()

        time.sleep(5)

    print("Simulación detenida.")





class TradingBotGUI:
    """Interfaz gráfica para controlar el bot de trading."""

    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot Control Panel")

        self.backtest_button = ttk.Button(root, text="Ejecutar Backtest", command=self.run_backtest)
        self.backtest_button.pack(pady=5)

        
        self.simulation_button = ttk.Button(root, text="Iniciar Simulación", command=self.start_simulation)
        self.simulation_button.pack(pady=5)

        # Etiqueta de estado
        self.status_label = ttk.Label(root, text="Estado: Inactivo", font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        # Botones principales
        self.start_button = ttk.Button(root, text="Iniciar Trading Real", command=self.start_trading)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(root, text="Detener Trading", command=self.stop_trading)  # Aquí se define correctamente
        self.stop_button.pack(pady=5)

        self.optimize_button = ttk.Button(root, text="Optimizar Modelos", command=self.optimize_models)
        self.optimize_button.pack(pady=5)

        self.evaluate_button = ttk.Button(root, text="Evaluar Modelos", command=self.evaluate_models)
        self.evaluate_button.pack(pady=5)

        self.stats_button = ttk.Button(root, text="Mostrar Estadísticas", command=self.show_statistics)
        self.stats_button.pack(pady=5)

        self.plot_button = ttk.Button(root, text="Mostrar Gráficos", command=self.show_plots)
        self.plot_button.pack(pady=5)

        self.exit_button = ttk.Button(root, text="Salir", command=root.quit)
        self.exit_button.pack(pady=10)

    def run_backtest(self):
        """Ejecuta el backtest solo cuando el usuario lo solicita."""
        self.status_label.config(text="Estado: Ejecutando Backtest")
        print("Ejecutando Backtest manualmente...")
        backtest()
        self.status_label.config(text="Estado: Backtest Completo")




    def start_trading(self):
        """Método para iniciar el trading en tiempo real."""
        global running
        #running = True
        self.status_label.config(text="Estado: Trading en tiempo real")
        print("Iniciando trading en tiempo real...")
        threading.Thread(target=real_trading, daemon=True).start()

    def start_simulation(self):
        """Método para iniciar la simulación sin bloquear la GUI."""
        global running
        if running:
            print("La simulación ya está en ejecución...")
            return

        running = True
        self.status_label.config(text="Estado: Simulación en Ejecución")
        print("Iniciando simulación en segundo plano...")

        # Iniciar la simulación en un hilo separado solo cuando se presiona el botón
        sim_thread = threading.Thread(target=simulation_thread, daemon=True)
        sim_thread.start()





    def run_simulation(self):
        """Ejecuta la simulación en segundo plano sin bloquear la interfaz."""
        global running

        if running:
            print("La simulación ya está en ejecución...")
            return

    running = True
        
    def stop_trading(self):
        """Método para detener el trading y la simulación."""
        global running
        running = False
        self.status_label.config(text="Estado: Trading detenido")
        print("Deteniendo trading y simulación...")


    def optimize_models(self):
        """Ejecuta la optimización de modelos de trading."""
        self.status_label.config(text="Estado: Optimizando Modelos")
        for symbol in SYMBOLS:
            df = get_market_data(symbol, my_candles)
            if df is not None:
                optimize_model(symbol, df)
        self.status_label.config(text="Estado: Optimización Completa")

    def evaluate_models(self):
        """Ejecuta la evaluación de modelos de trading."""
        self.status_label.config(text="Estado: Evaluando Modelos")

        for symbol in SYMBOLS:
            print(f"Intentando evaluar modelo para {symbol}...")
            df = get_market_data(symbol, my_candles)

            if df is not None and not df.empty:
                print(f"Evaluando modelo para {symbol}...")
                evaluate_model(symbol, df)
            else:
                print(f"Advertencia: No hay datos disponibles para evaluar {symbol}.")

        print("Evaluación de modelos finalizada.")
        self.status_label.config(text="Estado: Evaluación Completa")


        for symbol in SYMBOLS:
            print(f"Intentando evaluar modelo para {symbol}...")
            df = get_market_data(symbol, my_candles)

            if df is not None and not df.empty:
                print(f"Evaluando modelo para {symbol}...")
                evaluate_model(symbol, df)
            else:
                print(f"Advertencia: No hay datos disponibles para evaluar {symbol}.")
        
            print("Evaluación de modelos finalizada.")
            self.status_label.config(text="Estado: Evaluación Completa")


        for symbol in SYMBOLS:
            print(f"Evaluando modelo para {symbol}...")  # Verifica que cada modelo esté siendo evaluado
            df = get_market_data(symbol, my_candles)

            if df is not None and not df.empty:
                evaluate_model(symbol, df)
            else:
                print(f"Advertencia: No hay datos disponibles para evaluar {symbol}.")  # Verifica problemas de datos
        
        print("Evaluación de modelos finalizada.")  # Confirma que terminó correctamente
        self.status_label.config(text="Estado: Evaluación Completa")


    def show_statistics(self):
        """Muestra estadísticas del trading."""
        save_trading_statistics()
        messagebox.showinfo("Estadísticas", "Estadísticas actualizadas y guardadas en CSV.")

    def show_plots(self):
        """Abre una ventana con los gráficos de predicciones."""
        threading.Thread(target=plot_predictions, daemon=True).start()


def analyze_correlation():
    """Calcula y visualiza la correlación entre los activos seleccionados."""
    print("Calculando correlaciones entre activos...")

    market_data = {}
    for symbol in SYMBOLS:
        df = get_market_data(symbol, my_candles)
        if df is not None:
            market_data[symbol] = df['close']

    if len(market_data) < 2:
        print("No hay suficientes datos de activos para calcular la correlación.")
        return

    # Crear un DataFrame con los precios de cierre de cada activo
    df_correlation = pd.DataFrame(market_data)

    # Calcular la matriz de correlación
    correlation_matrix = df_correlation.corr()

    # Guardar la matriz de correlación en un archivo CSV
    correlation_matrix.to_csv("correlation_matrix.csv")
    print("Matriz de correlación guardada en 'correlation_matrix.csv'.")

    # Visualizar la correlación con un heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación entre Activos")
    plt.show()

def detect_trend(df):
    """Detecta la tendencia del mercado usando medias móviles y RSI."""

    if df is None or len(df) < 50:
        return "LATERAL"

    # Cálculo de medias móviles
    short_ma = df['close'].rolling(window=10).mean()
    long_ma = df['close'].rolling(window=50).mean()

    # Cálculo del RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Evitar división por cero
    loss.replace(0, np.nan, inplace=True)
    rs = gain / loss
    rs.fillna(0, inplace=True)
    rsi = 100 - (100 / (1 + rs))
    rsi.fillna(50, inplace=True)

    # Valores recientes de los indicadores
    short_ma_latest = short_ma.iloc[-1]
    long_ma_latest = long_ma.iloc[-1]
    rsi_latest = rsi.iloc[-1]

    # Ajustar los umbrales para evitar excesiva lateralidad
    ma_threshold = 0.001  # 0.1% de tolerancia en medias móviles
    rsi_up_threshold = 55
    rsi_down_threshold = 45

    # Condiciones para identificar la tendencia
    if short_ma_latest > long_ma_latest * (1 + ma_threshold) and rsi_latest > rsi_up_threshold:
        return "ALCISTA"
    elif short_ma_latest < long_ma_latest * (1 - ma_threshold) and rsi_latest < rsi_down_threshold:
        return "BAJISTA"
    else:
        return "LATERAL"





def get_economic_news():
    """Obtiene noticias económicas de alto impacto de una fuente de datos."""
    print("Obteniendo noticias económicas...")

    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"  # Fuente de datos de Forex Factory
    try:
        response = requests.get(url)
        news_data = response.json()
    except Exception as e:
        print(f"Error al obtener noticias económicas: {e}")
        return []

    high_impact_events = []
    for event in news_data:
        if event.get("impact") == "High":
            high_impact_events.append({
                "time": event.get("time"),
                "currency": event.get("country"),
                "event": event.get("title")
            })

    return high_impact_events

def validate_trade_signal(symbol, predicted_price, actual_price, df):
    """Valida si la predicción del modelo es confiable antes de ejecutar una orden."""
    
    if df is None or df.empty:
        print(f"Advertencia: Datos insuficientes para validar la señal de {symbol}.")
        return False

    # Verificación de variación significativa en la predicción
    price_change = abs((predicted_price - actual_price) / actual_price) * 100
    min_threshold = 0.15  # Umbral ajustado para detectar más oportunidades

    if price_change < min_threshold:
        print(f"Señal de {symbol} descartada: Variación menor al {min_threshold*100:.2f}% ({price_change:.5f}%)")
        return False

    # Confirmación con medias móviles
    if 'SMA_10' not in df.columns or 'SMA_50' not in df.columns:
        print(f"Error: Faltan columnas de medias móviles en {symbol}. No se puede validar la señal.")
        return False

    short_ma = df['SMA_10'].iloc[-1]
    long_ma = df['SMA_50'].iloc[-1]

    if predicted_price > actual_price and short_ma < long_ma:
        print(f"Señal de compra en {symbol} descartada: Media móvil corta debajo de la larga.")
        return False
    if predicted_price < actual_price and short_ma > long_ma:
        print(f"Señal de venta en {symbol} descartada: Media móvil corta encima de la larga.")
        return False

    # Confirmación con Volatilidad
    if 'volatility' not in df.columns:
        print(f"Error: Columna 'volatility' no encontrada en {symbol}. No se puede validar la señal.")
        return False

    volatility = df['volatility'].iloc[-1]
    max_volatility = 0.02  # Umbral de volatilidad ajustado al 2%

    if volatility > max_volatility:
        print(f"Señal de {symbol} descartada: Volatilidad alta ({volatility:.5f})")
        return False

    # Confirmación con RSI (Relative Strength Index)
    if 'RSI' in df.columns:
        rsi_latest = df['RSI'].iloc[-1]

        if predicted_price > actual_price and rsi_latest > 70:
            print(f"Señal de compra en {symbol} descartada: RSI en sobrecompra ({rsi_latest:.2f})")
            return False
        if predicted_price < actual_price and rsi_latest < 30:
            print(f"Señal de venta en {symbol} descartada: RSI en sobreventa ({rsi_latest:.2f})")
            return False
    else:
        print(f"Advertencia: RSI no encontrado en {symbol}. Se omitirá en la validación.")

    # Si pasa todas las validaciones, la señal es válida
    print(f"Señal de {symbol} validada: Predicción {predicted_price:.5f}, Precio actual {actual_price:.5f}")
    return True





def log_model_performance(symbol, predicted_price, actual_price, trend, volatility):
    """Registra el desempeño del modelo en un archivo CSV."""
    error = abs(predicted_price - actual_price)
    performance_data = [symbol, predicted_price, actual_price, error, trend, volatility, time.strftime('%Y-%m-%d %H:%M:%S')]

    filename = "model_performance_log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Símbolo", "Predicción", "Precio Real", "Error", "Tendencia", "Volatilidad", "Fecha"])
        writer.writerow(performance_data)

    print(f"Desempeño de {symbol} registrado: Error={error:.5f}, Tendencia={trend}, Volatilidad={volatility:.5f}")

def analyze_model_performance():
    """Analiza el rendimiento de los modelos y decide si necesitan ser reentrenados."""
    try:
        df = pd.read_csv("model_performance_log.csv")
    except FileNotFoundError:
        print("No hay datos de desempeño para analizar. Saltando optimización.")
        return

    avg_error = df.groupby("Símbolo")["Error"].mean()
    
    for symbol, error in avg_error.items():
        if error > 0.005:  # Si el error promedio es mayor al 0.5% del precio
            print(f"El modelo para {symbol} tiene un error alto ({error:.5f}). Se procederá a reentrenarlo.")
            retrain_model(symbol)



def retrain_model(symbol):
    """Reentrena el modelo asegurando que la normalización sea consistente con el entrenamiento."""
    global retraining_status

    if retraining_status.get(symbol, False):
        print(f"El modelo para {symbol} ya se está reentrenando. Esperando a que finalice...")
        return  

    retraining_status[symbol] = True  
    print(f"Reentrenando modelo para {symbol}...")

    # Cargar datos históricos
    historical_data_filename = f"market_data_{symbol}.csv"
    if os.path.exists(historical_data_filename):
        df_historical = pd.read_csv(historical_data_filename)
        df_historical['time'] = pd.to_datetime(df_historical['time'])
    else:
        df_historical = pd.DataFrame()

    # Obtener datos recientes del mercado
    df_recent = get_market_data(symbol, my_candles)
    if df_recent is None or df_recent.empty:
        print(f"No se pudo obtener datos recientes para reentrenar {symbol}.")
        retraining_status[symbol] = False
        return

    df_recent = add_technical_indicators(df_recent)
    df = pd.concat([df_historical, df_recent]).drop_duplicates(subset=['time']).reset_index(drop=True)

    features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']

    # Normalización correcta
    scaler_X = MinMaxScaler()
    df[features] = scaler_X.fit_transform(df[features])

    scaler_y = MinMaxScaler()
    df['close_scaled'] = scaler_y.fit_transform(df[['close']])

    # Generar datos de entrenamiento
    X, y = [], []
    for i in range(INPUT_TIMESTEPS, len(df)):
        X.append(df.iloc[i-INPUT_TIMESTEPS:i][features].values)
        y.append(df.iloc[i]['close_scaled'])  # Ahora usamos 'close_scaled'

    X, y = np.array(X), np.array(y)

    # Construcción del modelo LSTM
    model = keras.models.Sequential()

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True), input_shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))
    else:
        model.add(LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))

    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_RATE_1))

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False)))
    else:
        model.add(LSTM(LSTM_UNITS_2, return_sequences=False))

    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_RATE_2))

    model.add(Dense(DENSE_UNITS))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Guardar modelo y escaladores
    model.save(f"lstm_model_{symbol}.keras")
    models[symbol] = model
    joblib.dump(scaler_y, f"scaler_y_{symbol}.pkl")  # Guardar el escalador de y
    print(f"Modelo reentrenado y escalador guardado para {symbol}.")

    retraining_status[symbol] = False





def manage_trailing_stop():
    """Ajusta el Stop Loss dinámico con trailing stop y muestra los cambios en pantalla."""
    if not check_mt5_connection():
        return

    positions = mt5.positions_get()
    if not positions:
        return

    for position in positions:
        if position.magic != magicNumber:
            continue

        symbol = position.symbol
        order_type = position.type
        price_open = position.price_open
        current_price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        point = mt5.symbol_info(symbol).point
        new_sl = None

        # Cálculo del nuevo SL basado en trailing stop y trailing step
        if order_type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - TRAILING_STOP_PIPS * point
            if (new_sl - position.sl) > (TRAILING_STEP_PIPS * point):
                print(f"\nActualizando Trailing Stop en {symbol}")
                print(f"Nuevo SL: {new_sl:.5f} (Antes: {position.sl:.5f})")

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Trailing Stop actualizado en {symbol} a {new_sl:.5f}")

        elif order_type == mt5.ORDER_TYPE_SELL:
            new_sl = current_price + TRAILING_STOP_PIPS * point
            if (position.sl - new_sl) > (TRAILING_STEP_PIPS * point):
                print(f"\nActualizando Trailing Stop en {symbol}")
                print(f"Nuevo SL: {new_sl:.5f} (Antes: {position.sl:.5f})")

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Trailing Stop actualizado en {symbol} a {new_sl:.5f}")




# Ejecución del trading en tiempo real
def real_trading():
    """Ejecuta trading en tiempo real asegurando que las predicciones sean en la escala original."""
    global running

    if not models:
        print("Cargando modelos antes del Trading Real...")
        load_models()

    print("Iniciando ejecución en modo real...")
    running = True  

    while running:
        for symbol in SYMBOLS:
            print(f"\nProcesando {symbol} en tiempo real...")

            df = get_market_data(symbol, my_candles)
            if df is None or df.empty:
                print(f"Advertencia: No hay datos suficientes para {symbol}. Omitiendo...")
                continue

            df = add_technical_indicators(df)

            if symbol not in models:
                print(f"Error: Modelo para {symbol} no encontrado. Se recomienda regenerarlo.")
                continue

            try:
                required_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']
                X_input = np.array([df.iloc[-10:][required_columns].values])

                # Hacer predicción y desnormalizar
                predicted_scaled = models[symbol].predict(X_input)[0][0]

                # Cargar el escalador y desnormalizar la predicción
                scaler_y_filename = f"scaler_y_{symbol}.pkl"
                if os.path.exists(scaler_y_filename):
                    scaler_y = joblib.load(scaler_y_filename)
                    predicted_price = scaler_y.inverse_transform([[predicted_scaled]])[0][0]
                else:
                    print(f"Error: No se encontró el escalador de {symbol}. Usando predicción sin desnormalizar.")
                    predicted_price = predicted_scaled

                actual_price = df.iloc[-1]['close']

                print(f"Predicción para {symbol}: {predicted_price:.5f}, Precio actual: {actual_price:.5f}")

                # Guardar predicción
                save_prediction(symbol, predicted_price, actual_price)

                if not validate_trade_signal(symbol, predicted_price, actual_price, df):
                    print(f"Señal de {symbol} descartada tras validación.")
                    continue

                if check_existing_order(symbol):
                    print(f"Orden abierta detectada en {symbol}. No se enviará otra.")
                    continue

                order_type = "BUY" if predicted_price > actual_price else "SELL"
                print(f"Enviando orden {order_type} en {symbol} a precio {actual_price:.5f}...")
                send_order(symbol, order_type)

            except Exception as e:
                print(f"Error en el trading de {symbol}: {e}")

        check_orders()
        manage_trailing_stop()
        save_trading_statistics()
        time.sleep(5)

    print("Trading en tiempo real detenido.")













# Crear la ventana de la GUI
root = ttk.Tk()
app = TradingBotGUI(root)
root.mainloop()


# Registrar la función de pérdida personalizada en Keras
get_custom_objects()["custom_loss_function"] = custom_loss_function

if __name__ == "__main__":
    print("Esperando acciones del usuario en la interfaz gráfica...")
    root = ttk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()



