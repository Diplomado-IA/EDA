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
        rows.append(
            {
                "col": c,
                "dtype": str(s.dtype),
                "n_missing": n_missing,
                "pct_missing": round(pct_missing, 2),
                "n_unique": n_unique,
                "maybe_binary_text": _posible_binaria(s),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "col",
            "dtype",
            "n_missing",
            "pct_missing",
            "n_unique",
            "maybe_binary_text",
        ],
    )


def ordenar_resumen(df_resumen: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame de resumen por porcentaje de valores faltantes (desc)
    y cantidad de valores únicos (asc).

    Args:
        df_resumen: DataFrame retornado por resumen_columnas()

    Returns:
        DataFrame ordenado según los criterios especificados
    """
    # Para este caso específico, todos tienen pct_missing=0, entonces
    # estamos ordenando primariamente por n_unique ascendente
    return df_resumen.sort_values(
        by=["pct_missing", "n_unique"], ascending=[False, True]
    )


def obtener_top_faltantes(df_resumen: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """
    Devuelve las columnas con mayor porcentaje de valores faltantes.

    Args:
        df_resumen: DataFrame retornado por resumen_columnas()
        top: Cantidad de columnas a devolver (por defecto 10)

    Returns:
        DataFrame con las top columnas con más valores faltantes
    """
    return df_resumen.sort_values("pct_missing", ascending=False).head(top)
