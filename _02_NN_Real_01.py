import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

print("Buscando GPUs disponibles...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detectadas: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No se detectaron GPUs. Usando CPU.")

running = True
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "USDMXN"]
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
            print(f"Modelo no encontrado para {symbol}. Será generado en el futuro si es necesario.")

# Guardar predicciones en CSV
def save_predictions(symbol, prediction, actual_price):
    filename = f"predictions_realtime_{symbol}.csv"
    df = pd.DataFrame([[symbol, prediction, actual_price, time.time()]], columns=["symbol", "prediction", "actual_price", "timestamp"])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# Obtener datos de mercado
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
    return df

# Calcular beneficio real
def calculate_real_profit(symbol, lot_size, entry_price, sl_price):
    point = 0.1 if symbol == 'GDAXI' else mt5.symbol_info(symbol).point
    pip_value = point * 10 if symbol.endswith("JPY") else point * 100000
    swap = mt5.symbol_info(symbol).swap_long if entry_price < sl_price else mt5.symbol_info(symbol).swap_short
    commission = mt5.account_info().trade_commission if hasattr(mt5.account_info(), 'trade_commission') else 0
    profit = (sl_price - entry_price) * lot_size / point * pip_value
    return profit - swap - commission

# Enviar orden
def send_order(symbol, action, predicted_change):
    print(f"Intentando enviar orden {action} en {symbol}...")
    existing_orders = mt5.positions_get(symbol=symbol)
    if existing_orders and len(existing_orders) > 0:
        print(f"Ya existe una orden abierta en {symbol}, no se enviará otra.")
        return
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
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.1,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 123456,
        "comment": "AI Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Orden {action} ejecutada en {symbol}. Precio: {price}, Volumen: 0.1")
    else:
        print(f"Error al enviar orden en {symbol}: {result.comment}")

# Gestión de órdenes abiertas
def check_orders():
    for symbol in SYMBOLS:
        orders = mt5.positions_get(symbol=symbol)
        if orders is None:
            print(f"Error al obtener órdenes abiertas para {symbol}")
            continue
        orders = mt5.positions_get(symbol=symbol)
        


def update_trailing_stop(order, symbol):
    ticket = order.ticket
    order_type = order.type
    entry_price = order.price_open
    current_price = order.price_current
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"Error al obtener información de {symbol}")
        return
    
    point = symbol_info.point
    digits = symbol_info.digits
    pip_size = 10 ** (-digits + 1) if digits > 1 else point
    tick_size = point
    
    # Definir Stop y Step dinámicos
    stop_loss_pips = 50 * pip_size  # SL inicial ajustado dinámicamente
    step_pips = 10 * pip_size  # Step para mover el SL
    
    # Calcular nuevo SL basado en la dirección de la orden
    new_sl = entry_price + stop_loss_pips if order_type == 0 else entry_price - stop_loss_pips
    adjusted_sl = current_price - step_pips if order_type == 0 else current_price + step_pips
    
    profit = order.profit  # Definir correctamente el profit
    real_profit = calculate_real_profit(symbol, 0.1, entry_price, adjusted_sl)
    
    # Mover SL solo si hay beneficio suficiente y el nuevo SL es mejor
    if profit > 5 and real_profit > 5 and adjusted_sl > order.sl and ((order_type == 0 and adjusted_sl > entry_price) or (order_type == 1 and adjusted_sl < entry_price)):
        print(f"Moviendo SL en {symbol}: Anterior SL: {order.sl}, Nuevo SL: {adjusted_sl}, Beneficio: {real_profit}")
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": adjusted_sl,
            "tp": order.tp
        }
        modify_result = mt5.order_send(modify_request)
        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Trailing Stop ajustado en {symbol}. Nuevo SL: {adjusted_sl}")
    
    # Cierre de orden si la ganancia es alta
    if profit > 10:
        print(f"Cerrando orden en {symbol} con ganancia: {profit}, Real Profit: {real_profit}")
        mt5.Close(symbol)
        print(f"Orden en {symbol} cerrada con ganancia de {profit}.")
    ticket = order.ticket
    order_type = order.type
    entry_price = order.price_open
    current_price = order.price_current
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"Error al obtener información de {symbol}")
        return
    
    point = symbol_info.point
    digits = symbol_info.digits
    pip_size = 10 ** (-digits + 1) if digits > 1 else point
    tick_size = point
    
    # Definir Stop y Step dinámicos
    stop_loss_pips = 50 * pip_size  # SL inicial ajustado dinámicamente
    step_pips = 10 * pip_size  # Step para mover el SL
    
    # Calcular nuevo SL basado en la dirección de la orden
    new_sl = entry_price + stop_loss_pips if order_type == 0 else entry_price - stop_loss_pips
    adjusted_sl = current_price - step_pips if order_type == 0 else current_price + step_pips
    
    profit = order.profit  # Definir correctamente el profit
    real_profit = calculate_real_profit(symbol, 0.1, entry_price, adjusted_sl)
    
    # Mover SL solo si hay beneficio suficiente y el nuevo SL es mejor
    if profit > 5 and real_profit > 5 and adjusted_sl > order.sl and ((order_type == 0 and adjusted_sl > entry_price) or (order_type == 1 and adjusted_sl < entry_price)):
        print(f"Moviendo SL en {symbol}: Anterior SL: {order.sl}, Nuevo SL: {adjusted_sl}, Beneficio: {real_profit}")
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": adjusted_sl,
            "tp": order.tp
        }
        modify_result = mt5.order_send(modify_request)
        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Trailing Stop ajustado en {symbol}. Nuevo SL: {adjusted_sl}")
    
    # Cierre de orden si la ganancia es alta
    if profit > 10:
        print(f"Cerrando orden en {symbol} con ganancia: {profit}, Real Profit: {real_profit}")
        mt5.Close(symbol)
        print(f"Orden en {symbol} cerrada con ganancia de {profit}.")
    ticket = order.ticket
    order_type = order.type
    entry_price = order.price_open
    current_price = order.price_current
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"Error al obtener información de {symbol}")
        return
    
    point = symbol_info.point
    digits = symbol_info.digits
    pip_size = 10 ** (-digits + 1) if digits > 1 else point
    tick_size = point
    
    # Definir Stop y Step dinámicos
    stop_loss_pips = 50 * pip_size  # SL inicial ajustado dinámicamente
    step_pips = 10 * pip_size  # Step para mover el SL
    
    # Calcular nuevo SL basado en la dirección de la orden
    new_sl = entry_price + stop_loss_pips if order_type == 0 else entry_price - stop_loss_pips
    adjusted_sl = current_price - step_pips if order_type == 0 else current_price + step_pips
    
    profit = order.profit  # Definir correctamente el profit
    real_profit = calculate_real_profit(symbol, 0.1, entry_price, adjusted_sl)
    
    # Mover SL solo si hay beneficio suficiente y el nuevo SL es mejor
    if profit > 5 and real_profit > 5 and adjusted_sl > order.sl and ((order_type == 0 and adjusted_sl > entry_price) or (order_type == 1 and adjusted_sl < entry_price)):
        print(f"Moviendo SL en {symbol}: Anterior SL: {order.sl}, Nuevo SL: {adjusted_sl}, Beneficio: {real_profit}")
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": adjusted_sl,
            "tp": order.tp
        }
        modify_result = mt5.order_send(modify_request)
        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Trailing Stop ajustado en {symbol}. Nuevo SL: {adjusted_sl}")
    
    # Cierre de orden si la ganancia es alta
    if profit > 10:
        print(f"Cerrando orden en {symbol} con ganancia: {profit}, Real Profit: {real_profit}")
        mt5.Close(symbol)
        print(f"Orden en {symbol} cerrada con ganancia de {profit}.")
    ticket = order.ticket
    order_type = order.type
    entry_price = order.price_open
    current_price = order.price_current
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"Error al obtener información de {symbol}")
        return
    
    point = symbol_info.point
    digits = symbol_info.digits
    pip_size = 10 ** (-digits + 1) if digits > 1 else point
    tick_size = point
    
    trailing_pips = 15 * pip_size  # Ajuste dinámico basado en el tamaño del pip
    sl_distance = 50 * pip_size  # Ajuste de distancia de SL dinámico
    
    new_sl = entry_price + trailing_pips if order_type == 0 else entry_price - trailing_pips
    adjusted_sl = current_price - sl_distance if order_type == 0 else current_price + sl_distance
    
    profit = order.profit  # Definir profit correctamente
    real_profit = calculate_real_profit(symbol, 0.1, entry_price, adjusted_sl)
    
    if profit > 5 and real_profit > 5 and adjusted_sl > order.sl and ((order_type == 0 and adjusted_sl > entry_price) or (order_type == 1 and adjusted_sl < entry_price)):
        print(f"Moviendo SL en {symbol}: Anterior SL: {order.sl}, Nuevo SL: {adjusted_sl}, Beneficio: {real_profit}")
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": adjusted_sl,
            "tp": order.tp
        }
        modify_result = mt5.order_send(modify_request)
        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Trailing Stop ajustado en {symbol}. Nuevo SL: {adjusted_sl}")
    if profit > 10:
        print(f"Cerrando orden en {symbol} con ganancia: {profit}, Real Profit: {real_profit}")
        mt5.Close(symbol)
        print(f"Orden en {symbol} cerrada con ganancia de {profit}.")
    ticket = order.ticket
    order_type = order.type
    entry_price = order.price_open
    current_price = order.price_current
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    digits = symbol_info.digits
    pip_size = 10 ** (-digits + 1) if digits > 1 else point
    tick_size = point
    
    trailing_pips = 15 * pip_size  # Ajuste dinámico basado en el tamaño del pip
    sl_distance = 50 * pip_size  # Ajuste de distancia de SL dinámico
    
    new_sl = entry_price + trailing_pips if order_type == 0 else entry_price - trailing_pips
    adjusted_sl = current_price - sl_distance if order_type == 0 else current_price + sl_distance
    real_profit = calculate_real_profit(symbol, 0.1, entry_price, adjusted_sl)
    
    if order.profit > 5 and real_profit > 5 and adjusted_sl > order.sl and ((order.type == 0 and adjusted_sl > order.price_open) or (order.type == 1 and adjusted_sl < order.price_open)):
        print(f"Moviendo SL en {symbol}: Anterior SL: {order.sl}, Nuevo SL: {adjusted_sl}, Beneficio: {real_profit}")
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": adjusted_sl,
            "tp": order.tp
        }
        modify_result = mt5.order_send(modify_request)
        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Trailing Stop ajustado en {symbol}. Nuevo SL: {adjusted_sl}")
    if order.profit > 10:
        print(f"Cerrando orden en {symbol} con ganancia: {order.profit}, Real Profit: {real_profit}")
        mt5.Close(symbol)
        print(f"Orden en {symbol} cerrada con ganancia de {order.profit}.")
        return

# Ejecución del trading en tiempo real
def real_trading():
    print("Iniciando ejecución en modo real...")
    lookback = 10  # Debe coincidir con el número de pasos de entrada del modelo
    
    while running:
        for symbol in SYMBOLS:
            df = get_market_data(symbol, n_candles=lookback + 1)  # Asegurar suficientes datos
            if df is None or len(df) < lookback + 1:
                print(f"Datos insuficientes para {symbol}, omitiendo predicción.")
                continue
            
            # Seleccionar solo las columnas usadas durante el entrenamiento
            df_numeric = df[['open', 'high', 'low', 'close', 'tick_volume']].iloc[-lookback:]
            
            # Verificar dimensiones
            if df_numeric.shape[1] != 5:
                print(f"Dimensión incorrecta en datos de {symbol}: {df_numeric.shape}")
                continue
            
            # Convertir datos a numpy array con las dimensiones adecuadas
            X_input = np.array([df_numeric.values.astype(float)])
            
            # Hacer la predicción
            predicted_price = models[symbol].predict(X_input)[0][0]
            
            # Obtener el último precio real
            actual_price = df.iloc[-1]['close']
            
            # Guardar predicción
            save_predictions(symbol, predicted_price, actual_price)
            
            # Determinar si se debe comprar o vender
            action = "BUY" if predicted_price > actual_price else "SELL"
            send_order(symbol, action, abs(predicted_price - actual_price))
            
            # Verificar órdenes abiertas y trailing stop
            check_orders()
        
        time.sleep(5)


load_models()
real_trading()
