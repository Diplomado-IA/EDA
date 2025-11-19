"""Detección y mitigación de fuga/tautología del target."""
from typing import List, Dict, Any
import os
import json
import pandas as pd
from sklearn.linear_model import LinearRegression


def detect_leakage(df: pd.DataFrame, target: str, suspect_features: List[str], r2_threshold: float = 0.9) -> Dict[str, Any]:
    """Calcula R^2 del target respecto a features sospechosas; marca fuga si R^2 >= umbral."""
    feats = [f for f in suspect_features if f in df.columns and f != target]
    report = {"flag": False, "r2": None, "tested_features": feats}
    if not feats or target not in df.columns or len(df) < 3:
        return report
    X = df[feats].values
    y = df[target].values
    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))
    report["r2"] = r2
    report["flag"] = r2 >= r2_threshold
    return report


def drop_leaky_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Elimina columnas marcadas como fugas si existen."""
    to_drop = [f for f in features if f in df.columns]
    return df.drop(columns=to_drop, errors="ignore")


def save_leakage_report(report: Dict[str, Any], path: str = "reports/leakage_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
