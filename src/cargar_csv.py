# Función robusta para carga segura de csv
import pandas as pd
from pathlib import Path

def cargar_csv(ruta:str, sep:str=",", encoding_prioridad=("utf-8", "latin-1")) -> pd.DataFrame:
    """
    Lee un archivo CSV probando múltiples 'encodings' y valida la existencia del archivo.

    Parámetros:
    - ruta: str. Ruta al CSV (relativa o absoluta).
    - sep: str. Separador de columnas ("," para CSV estándar, ";" típico de Excel en español).
    - encoding_prioridad: tupla de encodings a probar en orden.

    Retorna:
    - pd.DataFrame con los datos cargados.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}" )

    ultimo_error = None
    for enc in encoding_prioridad:
        try:
            df = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)
            print(f"[OK] Cargado con encoding='{enc}' y separador='{sep}'. Filas={len(df)}, Columnas={df.shape[1]}" )
            return df
        except Exception as e:
            ultimo_error = e
            continue

    raise RuntimeError(f"No se pudo leer el CSV con encodings {encoding_prioridad}. Último error: {ultimo_error}")

# === USO DIDÁCTICO ===
df = cargar_csv("datos.csv")     # Cambia a sep=";" si tu CSV usa punto y coma
print(df.head(3))                 # Vista rápida de las primeras 3 filas
print(df.dtypes)                  # Tipos de cada columna
