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
#from keras_tuner import RandomSearch  # Solo lo mantenemos si realmente se usa
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
from sklearn.preprocessing import StandardScaler  # Asegurar que está importado
import logging
import joblib





magicNumber = 789654
my_candles = 2000 #5000
my_TIMEFRAME = mt5.TIMEFRAME_M5

# Configuración de modelo LSTM optimizado
USE_BIDIRECTIONAL = False #True  # Activar LSTM bidireccional
LSTM_UNITS_1 = 256  # Aumentamos la capacidad de la primera capa
LSTM_UNITS_2 = 128  # Segunda capa con más unidades
DROPOUT_RATE_1 = 0.05  # Reducimos el dropout para evitar perder demasiada información
DROPOUT_RATE_2 = 0.05
INPUT_TIMESTEPS = 30  # Aumentamos la ventana de observación
LEARNING_RATE = 0.0003  # Reducimos la tasa de aprendizaje para mayor estabilidad
INPUT_FEATURES = 9  # Mantener 9 features
DENSE_UNITS = 1  # Capa de salida con 1 neurona


# Hiperparámetros de entrenamiento optimizados
EPOCHS = 50 #200  # Aumentamos las épocas
BATCH_SIZE = 32 #16  # Reducimos batch size para mayor precisión

# Configuración de los parámetros de trading
SL_PIPS = 100 #500  # Stop Loss en pips
TP_PIPS = 200 #1000  # Take Profit en pips
TRAILING_STOP_PIPS = 100 #500  # Distancia mínima del SL respecto al precio actual
TRAILING_STEP_PIPS = 50 #200  # Pips que deben moverse antes de actualizar el SL



# Variables globales
models = {}  # Diccionario para almacenar modelos de ML
SYMBOLS = ["EURUSD"] #, "USDJPY", "USDMXN", "XAUUSD"]
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

def create_lstm_model():
    """Crea un modelo LSTM optimizado para trading con regularización para evitar sobreajuste."""
    model = keras.models.Sequential()
    model.add(Input(shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))))
    else:
        model.add(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Aumentamos dropout para evitar sobreajuste

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))))
    else:
        model.add(LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Agregamos una capa intermedia para mejorar la captación de patrones
    model.add(Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())

    model.add(Dense(1))  # Predecimos la diferencia con el precio actual, no el precio absoluto

    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")

    return model




def validate_and_save_model(model, symbol, X_test, y_true, scaler_y):
    """Evalúa el modelo antes de guardarlo y solo lo reemplaza si es mejor."""
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Evaluación del modelo {symbol}:")
    print(f"   - MSE: {mse:.5f}")
    print(f"   - MAE: {mae:.5f}")
    print(f"   - R²: {r2:.5f}")

    if r2 < 0:
        print(f"⚠ Advertencia: Modelo {symbol} tiene R² negativo. No se guardará.")
        return False

    model.save(f"lstm_model_{symbol}.keras")
    joblib.dump(scaler_y, f"scaler_y_{symbol}.pkl")
    print(f"✅ Modelo {symbol} guardado con éxito.")
    return True


def validate_trade_signal(symbol, predicted_price, actual_price, df):
    """Valida si la predicción es confiable antes de enviar una orden."""
    if df is None or df.empty:
        return False

    price_change = abs((predicted_price - actual_price) / actual_price) * 100

    # Si el cambio de precio es menor a 0.15%, ignorar la señal
    if price_change < 0.15:
        return False

    short_ma = df['SMA_10'].iloc[-1]
    long_ma = df['SMA_50'].iloc[-1]

    # Solo comprar si la media corta supera la larga
    if predicted_price > actual_price and short_ma < long_ma:
        return False

    # Solo vender si la media corta está por debajo de la larga
    if predicted_price < actual_price and short_ma > long_ma:
        return False

    # Verificar que el volumen sea alto para evitar señales falsas
    if df['tick_volume'].iloc[-1] < df['tick_volume'].rolling(50).mean().iloc[-1] * 0.8:
        return False

    # Validación con RSI para evitar operar en zonas extremas
    if 'RSI' in df.columns:
        rsi_latest = df['RSI'].iloc[-1]
        if predicted_price > actual_price and rsi_latest > 70:
            return False
        if predicted_price < actual_price and rsi_latest < 30:
            return False

    return True


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
# Deshabilitar GPU si no hay disponibles
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

# Opcional: Desactivar optimizaciones de oneDNN si causan inconsistencias
#import os
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



running = False

# Definir la función de pérdida antes de cargar modelos
def custom_loss_function(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

tf.keras.utils.get_custom_objects()["custom_loss_function"] = custom_loss_function


def train_lstm_model(X_train, y_train):
    """Entrena el modelo LSTM con Early Stopping para evitar sobreajuste."""
    model = create_lstm_model()

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                        epochs=100,  
                        batch_size=16,  
                        validation_split=0.2, 
                        callbacks=[early_stopping])

    return model


def validate_data(df):
    """Valida los datos antes del entrenamiento y corrige inconsistencias."""
    if df.isnull().sum().sum() > 0:
        print("Advertencia: Se encontraron valores NaN en los datos. Aplicando limpieza...")
        df.dropna(inplace=True)

    if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
        print("Error en los datos: Hay inconsistencias en los precios.")



retraining_status = {}  # Diccionario para controlar el estado del reentrenamiento

def retrain_model(symbol):
    """Reentrena el modelo LSTM para un símbolo específico integrando datos históricos desde CSV y nuevos datos del mercado."""
    global retraining_status

    if retraining_status.get(symbol, False):
        print(f"El modelo para {symbol} ya se está reentrenando. Esperando a que finalice...")
        return  

    retraining_status[symbol] = True  
    print(f"Reentrenando modelo para {symbol}...")

    historical_data_filename = f"historical_data_{symbol}.csv"

    if os.path.exists(historical_data_filename):
        df_historical = pd.read_csv(historical_data_filename)
        df_historical['time'] = pd.to_datetime(df_historical['time'])
    else:
        df_historical = pd.DataFrame()

    df_recent = get_market_data(symbol, my_candles)

    if df_recent is None or df_recent.empty:
        print(f"No se pudo obtener datos recientes para reentrenar {symbol}.")
        retraining_status[symbol] = False
        return

    df_recent = add_technical_indicators(df_recent)
    df = pd.concat([df_historical, df_recent]).drop_duplicates(subset=['time']).reset_index(drop=True)

    features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']

    # Aplicar normalización y guardar los escaladores
    scaler_X = MinMaxScaler()
    df[features] = scaler_X.fit_transform(df[features])

    scaler_y = MinMaxScaler()
    df['close_scaled'] = scaler_y.fit_transform(df[['close']])

    # Guardar escaladores después de normalizar los datos
    scaler_X_filename = f"scaler_X_{symbol}.pkl"
    scaler_y_filename = f"scaler_y_{symbol}.pkl"
    joblib.dump(scaler_X, scaler_X_filename)
    joblib.dump(scaler_y, scaler_y_filename)
    print(f"Escaladores guardados: {scaler_X_filename}, {scaler_y_filename}")

    X, y = [], []
    for i in range(INPUT_TIMESTEPS, len(df)):
        X.append(df.iloc[i-INPUT_TIMESTEPS:i][features].values)
        y.append(df.iloc[i]['close_scaled'])  

    X, y = np.array(X), np.array(y)

    # Definir el modelo
    model = keras.models.Sequential()
    model.add(Input(shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))  # Definir capa de entrada explícita

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True)))
    else:
        model.add(LSTM(LSTM_UNITS_1, return_sequences=True))

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

    # Entrenar el modelo
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Guardar el modelo entrenado
    model.save(f"lstm_model_{symbol}.keras")
    models[symbol] = model

    print(f"Modelo reentrenado y escalador guardado para {symbol}.")
    retraining_status[symbol] = False







# Verificar y cargar modelos existentes
def load_models():
    """Carga modelos LSTM y los reentrena si hay errores al cargarlos."""
    for symbol in SYMBOLS:
        model_path = f"lstm_model_{symbol}.keras"
        if os.path.exists(model_path):
            try:
                models[symbol] = load_model(model_path, custom_objects={"custom_loss_function": custom_loss_function})
            except Exception:
                retrain_model(symbol)
        else:
            retrain_model(symbol)





def add_technical_indicators(df):
    """Calcula indicadores técnicos con mejor manejo de datos NaN."""
    if df is None or df.empty:
        print("Error: No se puede calcular indicadores en un DataFrame vacío.")
        return df

    try:
        df['returns'] = df['close'].pct_change()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Corrección del error de pandas: Asignar correctamente sin `inplace=True`
        df.loc[:, 'RSI'] = df['RSI'].fillna(50)

        df['volatility'] = df['returns'].rolling(window=10).std()

        df.dropna(inplace=True)

    except Exception as e:
        print(f"Error al calcular indicadores técnicos: {e}")

    return df







def optimize_model(symbol, df):
    """Optimiza los hiperparámetros del modelo LSTM para mejorar su rendimiento."""
    print(f"Optimizando modelo para {symbol}...")

    # Asegurar que los indicadores están calculados antes de la normalización
    df = add_technical_indicators(df)

    # Definir las características esperadas
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']

    # Verificar si todas las características están en el DataFrame antes de aplicar la normalización
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print(f"Advertencia: Faltan las siguientes columnas en {symbol}: {missing_features}.")
        print(f"Revisando si los datos pueden calcularse nuevamente...")
        df = add_technical_indicators(df)  # Intentar calcular nuevamente los indicadores

        # Verificar si aún faltan columnas
        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            print(f"Error crítico: No se pueden encontrar las columnas necesarias {missing_features} en {symbol}.")
            return

    # Aplicar normalización de los datos
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Preparación de los datos
    X, y = [], []
    for i in range(10, len(df)):
        X.append(df.iloc[i-10:i][features].values)
        y.append(df.iloc[i]['close'])

    X, y = np.array(X), np.array(y)

    def build_model(hp):
        model = keras.models.Sequential()
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

    tuner.search(X, y, epochs=50, batch_size=16, validation_split=0.2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Mejores hiperparámetros para {symbol}: {best_hps.values}")

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)
    best_model.save(f"lstm_model_{symbol}.keras")

    models[symbol] = best_model
    print(f"Modelo optimizado para {symbol} guardado correctamente.")



def generate_model(symbol, df):
    """Genera un modelo LSTM o ejecuta la optimización de hiperparámetros si es necesario."""
    print(f"Generando modelo para {symbol}...")

    df = add_technical_indicators(df)

    model_path = f"lstm_model_{symbol}.keras"
    if not os.path.exists(model_path):
        print(f"No se encontró modelo para {symbol}. Iniciando optimización de hiperparámetros...")
        optimize_model(symbol, df)
    else:
        print(f"Modelo para {symbol} ya existe. Cargando...")
        models[symbol] = load_model(model_path)



def check_mt5_connection():
    """Verifica la conexión con MetaTrader 5 y la reinicia si es necesario."""
    if not mt5.initialize():
        log_message("Error al inicializar MT5. Intentando reiniciar...", "error")
        mt5.shutdown()
        time.sleep(2)
        if not mt5.initialize():
            log_message("Falló la reconexión a MT5. Verifica tu terminal.", "error")
            return False
    return True


def preprocess_and_evaluate(symbol):
    """Carga los indicadores técnicos antes de evaluar el modelo y verifica si el modelo existe."""
    print(f"Preparando datos para {symbol}...")

    # Verificar si el modelo está en memoria
    if symbol not in models:
        print(f"Modelo para {symbol} no encontrado en memoria. Intentando cargarlo...")
        load_models()  # Intenta cargar todos los modelos nuevamente

        if symbol not in models:  # Si sigue sin estar en memoria, hay un problema
            print(f"Error: Modelo para {symbol} no encontrado después de intentar cargarlo. Se recomienda reentrenarlo.")
            return

    # Obtener los datos del mercado
    df = get_market_data(symbol, my_candles)

    if df is None or df.empty:
        print(f"Advertencia: No hay datos suficientes para evaluar {symbol}.")
        return

    # Asegurar que los indicadores se calculan correctamente
    df = add_technical_indicators(df)

    # Evaluar el modelo con los datos procesados
    print(f"Evaluando modelo para {symbol}...")
    evaluate_model(symbol, df)



def evaluate_model(symbol, df):
    """Evalúa el modelo en datos recientes y usa predicciones previas si existen."""
    
    if df is None or df.empty:
        print(f"Advertencia: No hay datos suficientes para evaluar {symbol}.")
        return

    required_features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']

    scaler_X = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[required_features] = scaler_X.fit_transform(df_scaled[required_features])

    X_test, y_true = [], []
    for i in range(INPUT_TIMESTEPS, len(df_scaled)):
        X_test.append(df_scaled.iloc[i-INPUT_TIMESTEPS:i][required_features].values)
        y_true.append(df_scaled.iloc[i]['close'])

    X_test, y_true = np.array(X_test), np.array(y_true)

    model = models.get(symbol)
    if model is None:
        print(f"Error: Modelo para {symbol} no encontrado. Se recomienda reentrenarlo.")
        return

    y_pred_scaled = model.predict(X_test).flatten()

    scaler_y_filename = f"scaler_y_{symbol}.pkl"
    if os.path.exists(scaler_y_filename):
        scaler_y = joblib.load(scaler_y_filename)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    else:
        print(f"Advertencia: No se encontró el escalador de salida para {symbol}. Usando valores escalados.")
        y_pred = y_pred_scaled * (df['close'].max() - df['close'].min()) + df['close'].min()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Resultados de evaluación para {symbol}:")
    print(f"   - MSE: {mse:.5f}")
    print(f"   - MAE: {mae:.5f}")
    print(f"   - R²: {r2:.5f}")

    save_prediction(symbol, y_pred[-1], y_true[-1])

    if r2 < 0:
        print(f"Advertencia: R² negativo ({r2:.5f}) en {symbol}. Se recomienda reentrenar el modelo.")
        retrain_model(symbol)
    else:
        print(f"Evaluación completada para {symbol}. Modelo válido con R² de {r2:.5f}.")
















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
cached_market_data = {}

def get_market_data(symbol, n_candles, cache_time=30, save_to_csv=False):
    """Obtiene datos de mercado desde MT5 con caché y validaciones."""
    current_time = time.time()

    if symbol in cached_market_data:
        last_update, cached_df = cached_market_data[symbol]
        if current_time - last_update < cache_time:
            return cached_df

    if not check_mt5_connection():
        return None

    rates = mt5.copy_rates_from_pos(symbol, my_TIMEFRAME, 0, n_candles)
    if rates is None or len(rates) == 0:
        log_message(f"No se pudieron obtener datos para {symbol}.", "error")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.dropna(inplace=True)

    # Guardar en CSV si se solicita
    if save_to_csv:
        csv_filename = f"market_data_{symbol}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Datos de {symbol} guardados en {csv_filename}.")

    cached_market_data[symbol] = (current_time, df)
    return df











def backtest():
    """Ejecuta un backtest con validación de datos y carga modelos solo si no están cargados."""
    global models
    if not models:
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
                print(f"Error al cargar el modelo para {symbol}: {e}. Se procederá a reentrenarlo.")
                retrain_model(symbol)
        else:
            print(f"Modelo no encontrado para {symbol}. Se generará uno nuevo.")
            retrain_model(symbol)




def get_market_data_from_mt5(symbol, n_candles):
    """Obtiene datos en vivo desde MT5 y maneja errores."""
    print(f" Obteniendo datos en vivo para {symbol} desde MT5...")

    if not check_mt5_connection():
        return None

    rates = mt5.copy_rates_from_pos(symbol, my_TIMEFRAME, 0, n_candles)
    
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
    filename = "trading_statistics.csv"  # Definir el nombre del archivo correctamente

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

    # Guardar en el archivo correcto
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False, mode='w', encoding='utf-8')

    print(f"Estadísticas de trading guardadas en {filename}.")




class PaperTrading:
    """Simulador de trading sin ejecutar órdenes reales."""

    def __init__(self):
        self.balance = 10000  # Capital inicial ficticio
        self.positions = {}  # Almacena órdenes abiertas

    def send_order(symbol, action, lot_size=0.1):
        """Envía una orden de compra o venta en MT5 con detalles completos de SL, TP y trailing stop."""
        if not check_mt5_connection():
            return

        if check_existing_order(symbol):
            return

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        tick_info = mt5.symbol_info_tick(symbol)

        if tick_info is None or tick_info.ask is None or tick_info.bid is None:
            return

        price = tick_info.ask if action == "BUY" else tick_info.bid
        point = mt5.symbol_info(symbol).point

        if price is None or point is None:
            return

        sl = price - SL_PIPS * point if action == "BUY" else price + SL_PIPS * point
        tp = price + TP_PIPS * point if action == "BUY" else price - TP_PIPS * point

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
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log_message(f"Error al enviar orden en {symbol}: {result.comment}", "error")


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
                    print(f"orden simulada: {order_type} en {symbol} a {actual_price}...")
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

        self.backtest_button = ttk.Button(root, text="Entrenar Red Neuronal", command=self.run_backtest)
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
        """Ejecuta la evaluación de modelos de trading con preprocesamiento previo."""
        self.status_label.config(text="Estado: Evaluando Modelos")

        print("Cargando modelos antes de la evaluación...")
        load_models()  # Carga todos los modelos antes de evaluar

        for symbol in SYMBOLS:
            print(f"Intentando evaluar modelo para {symbol}...")
            preprocess_and_evaluate(symbol)  # Llama a la nueva función

            print("Evaluación de modelos finalizada.")
            self.status_label.config(text="Estado: Evaluación Completa")




    def show_statistics(self):
        """Muestra estadísticas del trading."""
        save_trading_statistics()
        messagebox.showinfo("Estadísticas", "Estadísticas actualizadas y guardadas en CSV.")

    def show_plots(self):
        """Abre una ventana con los gráficos de predicciones."""
        threading.Thread(target=plot_predictions, daemon=True).start()


def analyze_correlation():
    """Calcula la correlación entre los indicadores y el precio de cierre."""
    df = get_market_data("EURUSD", my_candles)
    df = add_technical_indicators(df)
    correlation_matrix = df.corr()
    print(correlation_matrix["close"])  # Ver relación con el precio de cierre




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
    """Verifica si la predicción es confiable antes de enviar una orden."""
    if df is None or df.empty:
        return False

    price_change = abs((predicted_price - actual_price) / actual_price) * 100
    if price_change < 0.15:
        return False

    short_ma = df['SMA_10'].iloc[-1]
    long_ma = df['SMA_50'].iloc[-1]
    
    if (predicted_price > actual_price and short_ma < long_ma) or \
       (predicted_price < actual_price and short_ma > long_ma):
        return False

    if df['volatility'].iloc[-1] > 0.02:
        return False

    if 'RSI' in df.columns:
        rsi_latest = df['RSI'].iloc[-1]
        if (predicted_price > actual_price and rsi_latest > 70) or \
           (predicted_price < actual_price and rsi_latest < 30):
            return False

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
    """Verifica si el modelo necesita ser reentrenado basándose en el error promedio."""
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
    """Reentrena el modelo LSTM para un símbolo específico integrando datos históricos desde CSV y nuevos datos del mercado."""
    global retraining_status

    if retraining_status.get(symbol, False):
        print(f"El modelo para {symbol} ya se está reentrenando. Esperando a que finalice...")
        return  

    retraining_status[symbol] = True  
    print(f"Reentrenando modelo para {symbol}...")

    historical_data_filename = f"historical_data_{symbol}.csv"

    if os.path.exists(historical_data_filename):
        df_historical = pd.read_csv(historical_data_filename)
        df_historical['time'] = pd.to_datetime(df_historical['time'])
    else:
        df_historical = pd.DataFrame()

    df_recent = get_market_data(symbol, my_candles)

    if df_recent is None or df_recent.empty:
        print(f"No se pudo obtener datos recientes para reentrenar {symbol}.")
        retraining_status[symbol] = False
        return

    df_recent = add_technical_indicators(df_recent)
    df = pd.concat([df_historical, df_recent]).drop_duplicates(subset=['time']).reset_index(drop=True)

    features = ['open', 'high', 'low', 'close', 'tick_volume', 'SMA_10', 'SMA_50', 'RSI', 'volatility']

    scaler_X = MinMaxScaler()
    df[features] = scaler_X.fit_transform(df[features])

    scaler_y = MinMaxScaler()
    df['close_scaled'] = scaler_y.fit_transform(df[['close']])

    X, y = [], []
    for i in range(INPUT_TIMESTEPS, len(df)):
        X.append(df.iloc[i-INPUT_TIMESTEPS:i][features].values)
        y.append(df.iloc[i]['close_scaled'])  

    X, y = np.array(X), np.array(y)

    # Definir el modelo
    model = keras.models.Sequential()
    model.add(Input(shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))  # Definir capa de entrada explícita

    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True)))
    else:
        model.add(LSTM(LSTM_UNITS_1, return_sequences=True))

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

    # Entrenar el modelo
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Guardar el modelo y el escalador de `close`
    model.save(f"lstm_model_{symbol}.keras")
    models[symbol] = model
    joblib.dump(scaler_y, f"scaler_y_{symbol}.pkl")  # Guardar el escalador de salida

    print(f"Modelo reentrenado y escalador guardado para {symbol}.")
    retraining_status[symbol] = False






def manage_trailing_stop():
    """Ajusta el Stop Loss dinámico si el precio avanza lo suficiente en beneficio."""
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
        point = mt5.symbol_info(symbol).point
        current_price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

        entry_price = position.price_open
        previous_sl = position.sl
        new_sl = previous_sl

        # Verifica si la posición está en beneficio antes de mover el SL
        if (order_type == mt5.ORDER_TYPE_BUY and current_price > entry_price) or \
           (order_type == mt5.ORDER_TYPE_SELL and current_price < entry_price):

            # Definir el nuevo SL si el precio ha avanzado el TRAILING_STEP_PIPS
            if order_type == mt5.ORDER_TYPE_BUY:
                if current_price - entry_price >= TRAILING_STEP_PIPS * point:
                    potential_sl = current_price - TRAILING_STOP_PIPS * point
                    if potential_sl > entry_price and (previous_sl < potential_sl or previous_sl <= entry_price):
                        new_sl = potential_sl

            elif order_type == mt5.ORDER_TYPE_SELL:
                if entry_price - current_price >= TRAILING_STEP_PIPS * point:
                    potential_sl = current_price + TRAILING_STOP_PIPS * point
                    if potential_sl < entry_price and (previous_sl > potential_sl or previous_sl >= entry_price):
                        new_sl = potential_sl

        # Solo actualizar el SL si hay un nuevo valor válido
        if new_sl != previous_sl:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Trailing Stop actualizado en {symbol} a {new_sl:.5f}")
            else:
                print(f"Error al actualizar Trailing Stop en {symbol}: {result.comment}")







# Ejecución del trading en tiempo real
def real_trading():
    """Ejecuta trading en tiempo real asegurando que las predicciones sean confiables y en la misma escala."""
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

                # Verificar que las columnas requeridas existen en los datos
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Error: Faltan las siguientes columnas en {symbol}: {missing_columns}")
                    continue

                # Cargar escaladores
                scaler_X_filename = f"scaler_X_{symbol}.pkl"
                scaler_y_filename = f"scaler_y_{symbol}.pkl"

                if os.path.exists(scaler_X_filename) and os.path.exists(scaler_y_filename):
                    scaler_X = joblib.load(scaler_X_filename)
                    scaler_y = joblib.load(scaler_y_filename)
                    df_scaled = pd.DataFrame(scaler_X.transform(df[required_columns]), columns=required_columns)
                else:
                    print(f"Advertencia: scaler_X o scaler_y no encontrados para {symbol}, usando datos sin escalar.")
                    df_scaled = df[required_columns]

                # Asegurar que se toman exactamente INPUT_TIMESTEPS
                if len(df_scaled) < INPUT_TIMESTEPS:
                    print(f"Advertencia: No hay suficientes datos recientes para {symbol}. Se requieren {INPUT_TIMESTEPS} timesteps.")
                    continue

                # Preparar entrada para la predicción
                X_input = np.array([df_scaled.iloc[-INPUT_TIMESTEPS:].values])

                # Realizar predicción
                predicted_scaled = models[symbol].predict(X_input)[0][0]
                predicted_price = scaler_y.inverse_transform([[predicted_scaled]])[0][0] if os.path.exists(scaler_y_filename) else predicted_scaled

                actual_price = df.iloc[-1]['close']

                print(f"Predicción para {symbol}: {predicted_price:.5f}, Precio actual: {actual_price:.5f}")

                # Guardar predicción
                save_prediction(symbol, predicted_price, actual_price)

                # Validar si la señal de trading es válida antes de operar
                if not validate_trade_signal(symbol, predicted_price, actual_price, df):
                    print(f"Señal de {symbol} descartada tras validación.")
                    continue

                if check_existing_order(symbol):
                    print(f"Orden abierta detectada en {symbol}. No se enviará otra.")
                    continue

                # Determinar si la orden será de compra o venta
                order_type = "BUY" if predicted_price > actual_price else "SELL"
                print(f"Enviando orden {order_type} en {symbol} a precio {actual_price:.5f}...")
                send_order(symbol, order_type)

            except Exception as e:
                print(f"Error en el trading de {symbol}: {e}")

        # Análisis de rendimiento del modelo
        analyze_model_performance()
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



