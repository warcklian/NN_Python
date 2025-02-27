import pandas as pd

df_metrics = pd.read_csv("model_evaluation.csv")
print(df_metrics.tail(10))  # Muestra las últimas 10 evaluaciones
