#!/usr/bin/env python3
"""Script para validar HU1 (fuga/tautología) sobre el target de regresión.
Uso:
  python scripts/run_regression_leakage.py [--strategy drop_features|redefine_target|fail]
Pasos:
  1) Carga dataset y aplica preprocess_pipeline (que incluye detección/mitigación de fuga).
  2) Entrena un RandomForestRegressor simple sobre X_train_engineered / y_train.
  3) Muestra artefactos clave: reports/leakage_report.json, leakage_action.txt y (si redefine) outputs/metadata/target_mapping.json.
"""
import os
import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import sys, os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.config import Config
from src.eda import cargar_csv
from src.preprocessing.clean import preprocess_pipeline

PARAMS_YAML = Path("config/params.yaml")


def load_params():
    try:
        import yaml  # type: ignore
        with open(PARAMS_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def override_strategy(strategy: str | None):
    if not strategy:
        return
    params = load_params()
    if "leakage" not in params:
        params["leakage"] = {}
    params["leakage"]["strategy"] = strategy
    try:
        import yaml  # type: ignore
        with open(PARAMS_YAML, "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f, allow_unicode=True)
    except Exception as e:
        print(f"[WARN] No se pudo sobreescribir strategy en params.yaml: {e}")


def main():
    ap = argparse.ArgumentParser(description="Valida fuga en target de regresión")
    ap.add_argument("--strategy", type=str, default=None, help="Override de leakage.strategy")
    args = ap.parse_args()

    override_strategy(args.strategy)
    params = load_params()
    print(f"Strategy actual: {params.get('leakage', {}).get('strategy')}")

    # Cargar dataset
    df = cargar_csv(Config.DATASET_PATH)
    # Normalizar nombres (strip) para detección robusta
    # No normalizar: usar nombres exactos con espacios finales
    # df.columns = [c.rstrip() for c in df.columns]
    target = Config.TARGET_REGRESSION
    if target not in df.columns:
        raise KeyError(f"Target de regresión '{target}' no encontrado. Columnas disponibles: {df.columns[-10:]} ...")
    print(f"Target regresión usado: '{target}'")

    # Ejecutar preproceso completo (incluye HU1)
    artifacts = preprocess_pipeline(df)
    print("Preprocess artifacts:")
    for k,v in artifacts.items():
        print(f"  {k}: {v}")

    # Cargar leakage report si existe
    rep_path = Path("reports/leakage_report.json")
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        print("\nLeakage report:")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print("\n[INFO] No se generó leakage_report.json (posible: target ausente o detect=false)")

    action_path = Path("reports/leakage_action.txt")
    if action_path.exists():
        print("\nLeakage action:")
        print(action_path.read_text(encoding="utf-8"))
    else:
        print("\n[INFO] No se generó leakage_action.txt (no hubo fuga o detect=false)")

    mapping_path = Path("outputs/metadata/target_mapping.json")
    if mapping_path.exists():
        print("\nTarget mapping (redefine_target):")
        print(mapping_path.read_text(encoding="utf-8"))

    # Entrenamiento simple (solo si no fail-fast)
    x_train_path = Path(artifacts.get("X_train_engineered", "data/processed/X_train_engineered.csv"))
    if not x_train_path.exists():
        print("[WARN] No existe X_train_engineered; posible fail-fast por fuga")
        return 0
    x_test_path = Path(artifacts.get("X_test_engineered", "data/processed/X_test_engineered.csv"))
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # Recuperar columna y_train del train original (tras ingeniería); usamos el target redefinido si aplica
    # Buscar columna de target (original, redefinido o residual)
    residual_col = None
    if mapping_path.exists():
        m = json.loads(mapping_path.read_text(encoding="utf-8"))
        residual_col = m.get("redefined_target")
    y_source_cols = [c for c in df.columns if c.strip() == target]
    if residual_col and residual_col in df.columns:
        # Recalcular y_train de train_df original filtrando años de entrenamiento
        # Tomamos años <= TRAIN_END_YEAR
        year_col = next((c for c in df.columns if c.strip().upper() == "AÑO"), None)
        if year_col:
            years = pd.to_numeric(df[year_col].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
            mask_train = (years <= Config.TRAIN_END_YEAR)
            y_train = df.loc[mask_train, residual_col]
        else:
            y_train = df[residual_col]
        print(f"Usando target residual para entrenamiento: {residual_col}")
    else:
        # Fallback: intentar cargar y desde archivo engineered si aún estaba presente
        # (en pipeline se eliminan columnas target de X)
        y_train = None
        print("[INFO] Entrenamiento omitido de regresión (target no integrado en engineered set). HU1 validada solo por fuga.")
        return 0

    # Alinear longitud de y_train con X_train (ambas deberían corresponder a train split)
    if len(y_train) != len(X_train):
        y_train = y_train.iloc[:len(X_train)]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("\nModelo regresión entrenado sobre residual target. Feature count:", X_train.shape[1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
