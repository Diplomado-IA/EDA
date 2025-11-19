#!/usr/bin/env python3
"""
Script de ejemplo para ejecutar HPO (GridSearchCV) usando la configuración del proyecto.
- Puede leer un CSV externo o usar el dataset California Housing como demo si no se pasa --data.
- Respeta CV temporal si cv.kind == "time" (requiere --date-col para ordenar).
"""
import argparse
import os
import sys
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.splits import get_cv
from src.models.train import run_grid_search
from src.features.leakage import detect_leakage, save_leakage_report


def load_config(path: str = "config/params.yaml") -> dict:
    cfg = {}
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        # Fallback mínimo si no hay PyYAML o archivo
        cfg = {
            "hpo": {
                "method": "grid",
                "metric": "neg_root_mean_squared_error",
                "random_state": 42,
                "grid": {
                    "n_estimators": [200, 400, 800],
                    "max_depth": [None, 6, 10, 14],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "cv": {"kind": "kfold", "n_splits": 5, "shuffle": False},
            "leakage": {
                "detect": True,
                "r2_threshold": 0.90,
                "strategy": "drop_features",
                "suspect_features": ["PROMEDIO_EDAD_HOMBRE", "PROMEDIO_EDAD_MUJER"],
                "target_base": "PROMEDIO_EDAD_BASE",
            },
        }
    return cfg


def prepare_data(args, cfg):
    if args.data:
        df = pd.read_csv(args.data)
        assert args.target, "Debe especificar --target cuando se usa --data"
        target = args.target
        # Orden temporal si aplica
        if cfg.get("cv", {}).get("kind") == "time" and args.date_col:
            if args.date_col not in df.columns:
                raise ValueError(f"date-col {args.date_col} no existe en datos")
            df = df.sort_values(args.date_col)
        y = df[target]
        X = df.drop(columns=[target])
        # Selección numérica simple y relleno básico
        X = X.select_dtypes(include=[np.number]).copy()
        X = X.fillna(X.median(numeric_only=True))
        return X, y, df
    # Demo: California Housing
    from sklearn.datasets import fetch_california_housing

    ds = fetch_california_housing()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, None


def maybe_check_leakage(df, target, cfg):
    rep = None
    if df is not None and cfg.get("leakage", {}).get("detect", True):
        suspects = cfg.get("leakage", {}).get("suspect_features", [])
        r2_thr = cfg.get("leakage", {}).get("r2_threshold", 0.9)
        rep = detect_leakage(df, target, suspects, r2_thr)
        save_leakage_report(rep)
    return rep


def build_estimator(cfg):
    rs = cfg.get("hpo", {}).get("random_state", 42)
    return RandomForestRegressor(random_state=rs, n_estimators=200)


def build_param_grid(cfg):
    grid = cfg.get("hpo", {}).get("grid", {})
    # Param names mapeados al estimador por defecto (RandomForestRegressor)
    return {
        "n_estimators": grid.get("n_estimators", [200, 400, 800]),
        "max_depth": grid.get("max_depth", [None, 6, 10, 14]),
        "min_samples_split": grid.get("min_samples_split", [2, 5, 10]),
        "min_samples_leaf": grid.get("min_samples_leaf", [1, 2, 4]),
    }


def persist_run_metadata(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sha = None
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        pass
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": sha,
    }
    md_dir = os.path.join("outputs", "metadata")
    os.makedirs(md_dir, exist_ok=True)
    with open(os.path.join(md_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Ejecuta HPO con GridSearchCV")
    parser.add_argument("--data", type=str, default=None, help="Ruta a CSV opcional")
    parser.add_argument("--target", type=str, default=None, help="Columna target si se pasa --data")
    parser.add_argument("--date-col", type=str, default=None, help="Columna temporal para ordenar si CV temporal")
    parser.add_argument("--out-dir", type=str, default="outputs/hpo", help="Directorio de salida para resultados HPO")
    parser.add_argument("--no-leak-check", action="store_true", help="Desactiva chequeo de fuga")
    args = parser.parse_args()

    cfg = load_config()

    X, y, df_raw = prepare_data(args, cfg)

    # Chequeo de fuga opcional (solo si tenemos df crudo)
    if not args.no_leak_check and df_raw is not None and args.target:
        maybe_check_leakage(df_raw, args.target, cfg)

    # CV
    cv = get_cv(cfg.get("cv", {}))

    # Estimador y grid
    estimator = build_estimator(cfg)
    param_grid = build_param_grid(cfg)

    # Métricas
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

    # Run HPO
    run_grid_search(estimator, param_grid, X, y, cv, scoring=scoring, out_dir=args.out_dir)

    # Persistir metadatos
    persist_run_metadata(args.out_dir)

    print(f"HPO finalizado. Resultados en {args.out_dir}")


if __name__ == "__main__":
    main()
