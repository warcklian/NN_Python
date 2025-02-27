import os
import glob

def clean_previous_data():
    """Elimina todos los archivos de datos previos, modelos y logs."""
    files_to_delete = glob.glob("market_data_*.csv") + \
                      glob.glob("predictions_*.csv") + \
                      glob.glob("lstm_model_*.keras") + \
                      glob.glob("scaler_y_*.pkl") + \
                      ["trading_statistics.csv", "model_performance_log.csv"]

    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Archivo eliminado: {file}")
        except Exception as e:
            print(f"No se pudo eliminar {file}: {e}")

    print("Todos los archivos previos han sido eliminados. El sistema está listo para regenerarse desde cero.")

clean_previous_data()
