# src/resumen_columnas.py
import numpy as np
import pandas as pd

def _posible_binaria(col: pd.Series) -> bool:
    """
    Heurística simple para columnas de texto que podrían ser binarias.
    """
    # Excluir NaN para conteo de valores únicos
    valores = col.dropna().unique()
    if len(valores) != 2:
        return False
    # Si son strings o categorías con 2 valores distintos
    if col.dtype == "object" or str(col.dtype).startswith("category"):
        return True
    return False

def resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con métricas de calidad por columna.
    """
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n * 100.0) if n > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))
        rows.append({
            "col": c,
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 2),
            "n_unique": n_unique,
            "maybe_binary_text": _posible_binaria(s),
        })
    return pd.DataFrame(rows, columns=["col","dtype","n_missing","pct_missing","n_unique","maybe_binary_text"])
