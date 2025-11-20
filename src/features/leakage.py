"""Detección y mitigación de fuga/tautología del target."""
from typing import List, Dict, Any
import os
import json
import pandas as pd
from sklearn.linear_model import LinearRegression


def detect_leakage(df: pd.DataFrame, target: str, suspect_features: List[str], r2_threshold: float = 0.9) -> Dict[str, Any]:
    """Calcula R^2 del target respecto a features sospechosas; robusto a tipos y valores.
    Pasos:
      1) Filtra features existentes y distintas del target.
      2) Coerce numérico (errores->NaN) para target y features.
      3) Elimina filas con NaN en cualquiera de las columnas involucradas.
      4) Si quedan <3 filas o todas las columnas son constantes, retorna sin marcar fuga pero registra razón.
      5) Ajusta regresión lineal y calcula R2. Si falla, intenta correlación promedio como proxy.
    """
    feats = [f for f in suspect_features if f in df.columns and f != target]
    report: Dict[str, Any] = {"flag": False, "r2": None, "tested_features": feats, "rows_used": None, "reason": None}
    if not feats:
        report["reason"] = "no_suspect_features_present"
        return report
    if target not in df.columns:
        report["reason"] = "target_missing"
        return report
    # Coerce numeric
    work = df[[target] + feats].copy()
    # Limpieza previa: strip, reemplazo de coma decimal y eliminación de espacios internos
    for c in work.columns:
        work[c] = (
            work[c]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .str.replace(r"\s+", "", regex=True)
        )
        # Coerción numérica final
        work[c] = pd.to_numeric(work[c], errors="coerce")
    # Drop rows with NaN
    work = work.dropna(axis=0, how="any")
    report["rows_used"] = len(work)
    if len(work) < 3:
        report["reason"] = "insufficient_rows_after_clean"
        return report
    # Check constant columns
    const_cols = [c for c in work.columns if work[c].nunique() <= 1]
    if const_cols:
        report["reason"] = f"constant_columns:{const_cols}"
        # If target is constant and any suspect equals target, treat as leakage (trivial)
        if target in const_cols and any(f in const_cols for f in feats):
            report["r2"] = 1.0
            report["flag"] = True
        return report
    X = work[feats].values
    y = work[target].values
    try:
        model = LinearRegression()
        model.fit(X, y)
        r2 = float(model.score(X, y))
        report["r2"] = r2
        report["flag"] = r2 >= r2_threshold
    except Exception as e:
        # Fallback: mean absolute correlation squared as proxy
        try:
            import numpy as np
            corr_mat = np.corrcoef(work.T)
            # target index 0, compute mean abs corr to suspects
            corrs = np.abs(corr_mat[0, 1:])
            proxy_r2 = float(np.mean(corrs) ** 2)
            report["r2"] = proxy_r2
            report["flag"] = proxy_r2 >= r2_threshold
            report["reason"] = f"linreg_failed:{e}"
        except Exception as e2:
            report["reason"] = f"linreg_and_proxy_failed:{e}|{e2}"
    return report


def drop_leaky_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Elimina columnas marcadas como fugas si existen."""
    to_drop = [f for f in features if f in df.columns]
    return df.drop(columns=to_drop, errors="ignore")


def save_leakage_report(report: Dict[str, Any], path: str = "reports/leakage_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
