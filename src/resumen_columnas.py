import pandas as pd
import numpy as np

def resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n * 100.0) if n else 0.0
        n_unique = int(s.nunique(dropna=True))
        rows.append({
            "col": c,
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 2),
            "n_unique": n_unique,
        })
    return pd.DataFrame(rows)

def ordenar_resumen(df_resumen: pd.DataFrame) -> pd.DataFrame:
    return df_resumen.sort_values(by=["pct_missing", "n_unique"], ascending=[False, True])

def obtener_top_faltantes(df_resumen: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    return df_resumen.sort_values("pct_missing", ascending=False).head(top)
