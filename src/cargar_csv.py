# src/cargar_csv.py
from pathlib import Path
import pandas as pd


def cargar_csv(
    ruta: str, sep: str = ",", encoding_prioridad=("utf-8", "latin-1")
) -> tuple[pd.DataFrame, dict]:
    """
    Lee un CSV de forma robusta probando varios encodings.

    Returns:
        Tupla con (dataframe, metadata_dict) donde metadata_dict contiene
        información de la carga.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}")

    ultimo_error = None
    for enc in encoding_prioridad:
        try:
            df = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)

            # Metadata para persistencia
            metadata = {
                "ruta_absoluta": str(ruta.resolve()),
                "nombre_archivo": ruta.name,
                "encoding_usado": enc,
                "separador": sep,
                "filas": len(df),
                "columnas": len(df.columns),
                "columnas_lista": list(df.columns),
                "tipos_columnas": {
                    col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)
                },
            }

            print(
                f"[OK] Cargado con encoding='{enc}', filas={len(df)}, columnas={len(df.columns)}"
            )
            return df, metadata
        except Exception as e:
            print(f"[WARN] Falló con encoding='{enc}': {e}")
            ultimo_error = e

    raise RuntimeError(
        f"No se pudo leer el CSV con los encodings {encoding_prioridad}"
    ) from ultimo_error
