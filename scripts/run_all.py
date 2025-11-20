#!/usr/bin/env python3
"""
Runner único: ejecuta el flujo completo (EDA -> Preproceso -> Entrenamiento -> Evaluación -> XAI)
Uso:
  python scripts/run_all.py [--stop-on-error] [--skip-train] [--skip-eval] [--skip-xai]
Notas:
  - Establece PYTHONPATH al raíz del proyecto para resolver imports (src/*).
  - Ejecuta condicionalmente cada paso si el script existe.
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STEPS = []  # Consolidado: pipeline se ejecuta inline


def run_step(name: str, cmd: list[str], desc: str, env: dict, stop_on_error: bool, skip: bool) -> bool:
    # Detectar ruta del script .py dentro del comando
    script_path = None
    for c in cmd:
        if isinstance(c, str) and c.endswith('.py'):
            p = Path(c)
            script_path = p if p.is_absolute() else (PROJECT_ROOT / p)
            break
    if skip:
        print(f"[SKIP] {name}: omitiendo por bandera")
        return True
    if script_path is not None and not script_path.exists():
        print(f"[SKIP] {name}: {script_path.name} no existe")
        return True
    print("\n" + "="*80)
    print(f"➡️  {desc}")
    print("="*80)
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
        if res.returncode != 0:
            print(f"✖ {name} falló con código {res.returncode}")
            if stop_on_error:
                return False
        else:
            print(f"✔ {name} completado")
        return True
    except Exception as e:
        print(f"✖ Error ejecutando {name}: {e}")
        return not stop_on_error


def list_artifacts():
    print("\n=== Artifacts (data/processed) ===")
    dp = PROJECT_ROOT / "data" / "processed"
    if dp.exists():
        for p in sorted(dp.rglob("*")):
            if p.is_file():
                try:
                    sz = p.stat().st_size // 1024
                    print(f"{p.relative_to(PROJECT_ROOT)}\t{sz} KB")
                except Exception:
                    pass
    print("\n=== Artifacts (outputs, models, reports) ===")
    for base in [PROJECT_ROOT/"outputs", PROJECT_ROOT/"models", PROJECT_ROOT/"reports"]:
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    try:
                        sz = p.stat().st_size // 1024
                        print(f"{p.relative_to(PROJECT_ROOT)}\t{sz} KB")
                    except Exception:
                        pass


def main():
    stop_on_error = "--stop-on-error" in sys.argv
    skip_train = "--skip-train" in sys.argv
    skip_eval = "--skip-eval" in sys.argv
    skip_xai = "--skip-xai" in sys.argv
    with_hpo = "--with-hpo" in sys.argv
    hpo_method = "grid"
    for arg in sys.argv:
        if arg.startswith("--hpo-method="):
            hpo_method = arg.split("=",1)[1].strip().lower()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    ok = True

    # 0) Fases 1-2: EDA y artefactos iniciales
    ok = run_step(
        name="EDA",
        cmd=[sys.executable, "scripts/execute_pipeline.py", "--phase", "all"],
        desc="Ejecución Fases 1-2 (EDA)",
        env=env,
        stop_on_error=stop_on_error,
        skip=False,
    ) and ok
    if not ok and stop_on_error:
        list_artifacts()
        print("\n⚠️ Flujo detenido por error en EDA")
        return 1

    # 1) Preproceso completo para generar artefactos (X_train/X_test engineered)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from config.config import Config
        from src.eda import cargar_csv
        from src.preprocessing.clean import _ensure_modalidad_bin, _coerce_regression_target, impute_values, temporal_split, preprocess_pipeline
        import pandas as pd
        from pathlib import Path as _Path

        print("\n" + "="*80)
        print("➡️  Ejecutando preprocesamiento (Fase 3)")
        print("="*80)
        df = cargar_csv(Config.DATASET_PATH)
        df = _ensure_modalidad_bin(df)
        df = _coerce_regression_target(df)
        df = impute_values(df)
        # y_train/y_test a partir del mismo split temporal
        train_df, test_df = temporal_split(df)
        task = "classification" if Config.TARGET_CLASSIFICATION in train_df.columns else "regression"
        y_col = Config.TARGET_CLASSIFICATION if task == "classification" else Config.TARGET_REGRESSION
        y_train = train_df[y_col]
        y_test = test_df[y_col]
        # Ejecutar pipeline para generar X engineered y artefactos
        artifacts = preprocess_pipeline(df)
        xtr_path = _Path(artifacts.get("X_train_engineered", "data/processed/X_train_engineered.csv"))
        xte_path = _Path(artifacts.get("X_test_engineered", "data/processed/X_test_engineered.csv"))
        X_train = pd.read_csv(xtr_path)
        X_test = pd.read_csv(xte_path)
        print(f"✔ Preproceso OK | X_train={X_train.shape} X_test={X_test.shape} | y: {y_col}")
        # Mostrar artefactos de fuga si existen
        import json as _json
        from pathlib import Path as _P
        leak_rep = _P("reports/leakage_report.json")
        leak_act = _P("reports/leakage_action.txt")
        if leak_rep.exists():
            try:
                rep = _json.loads(leak_rep.read_text(encoding="utf-8"))
                print(f"[LEAKAGE] flag={rep.get('flag')} R2={rep.get('r2')} tested={rep.get('tested_features')}")
            except Exception:
                print("[LEAKAGE] No se pudo leer leakage_report.json")
        if leak_act.exists():
            print("[LEAKAGE] acción aplicada:")
            print(leak_act.read_text(encoding="utf-8"))
        else:
            print("[LEAKAGE] sin acción (flag=false o estrategia no disparada)")
        # Guardar resumen rápido
        try:
            (_P("reports/leakage_summary.txt")).write_text(f"flag={rep.get('flag')} R2={rep.get('r2')} strategy={rep.get('strategy')}\n", encoding="utf-8")
        except Exception:
            pass
    except Exception as e:
        print(f"✖ preproceso falló: {e}")
        ok = False
        list_artifacts()
        print("\n" + ("✅ Flujo completado" if ok else "⚠️ Flujo completado con errores"))
        return 1

    # 2) Entrenamiento y evaluación
    model = None
    feature_names = list(X_train.columns)
    try:
        if not skip_train:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            print("\n" + "="*80)
            print("➡️  Entrenando modelo demo")
            print("="*80)
            best_params = None
            if with_hpo:
                hpo_out = f"outputs/hpo_{'clf' if task=='classification' else 'reg'}"
                os.makedirs(hpo_out, exist_ok=True)
                task_flag = ['--task', 'clf' if task=='classification' else 'reg']
                method_flag = ['--method', hpo_method]
                print(f"\n➡  Ejecutando HPO ({hpo_method}) antes del entrenamiento principal...")
                subprocess.run([sys.executable, 'scripts/run_hpo.py', *task_flag, *method_flag, '--fast', '--out-dir', hpo_out, '--no-leak-check'], cwd=PROJECT_ROOT, check=False)
                import json as _json
                best_file = Path(hpo_out)/'best.json'
                if best_file.exists():
                    try:
                        best_data = _json.loads(best_file.read_text(encoding='utf-8'))
                        best_params = best_data.get('best_params_', {})
                        print(f"[HPO] best_params={best_params}")
                    except Exception as e:
                        print(f"[HPO] No se pudo leer best.json: {e}")
            if task == "classification":
                if best_params:
                    model = RandomForestClassifier(random_state=42, n_jobs=-1, **{k:v for k,v in best_params.items() if k in {"n_estimators","max_depth","min_samples_split","min_samples_leaf"}})
                else:
                    model = RandomForestClassifier(n_estimators=200, max_depth=18, random_state=42, n_jobs=-1)
            else:
                if best_params:
                    model = RandomForestRegressor(random_state=42, n_jobs=-1, **{k:v for k,v in best_params.items() if k in {"n_estimators","max_depth","min_samples_split","min_samples_leaf"}})
                else:
                    model = RandomForestRegressor(n_estimators=200, max_depth=18, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            print("✔ Entrenamiento completado (con HPO" + (" aplicado" if best_params else " por defecto") + ")")
        else:
            print("[SKIP] Entrenamiento")
        if not skip_eval and model is not None:
            from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error
            print("\n" + "="*80)
            print("➡️  Evaluación en test")
            print("="*80)
            if task == "classification":
                try:
                    proba = model.predict_proba(X_test)[:,1]
                    auc = roc_auc_score(y_test, proba)
                    print(f"AUC-ROC: {auc:.3f}")
                    (_Path("reports").mkdir(parents=True, exist_ok=True))
                    (_Path("reports/metrics_classification.txt")).write_text(f"AUC-ROC={auc:.4f}\n", encoding="utf-8")
                except Exception:
                    pred = model.predict(X_test)
                    f1 = f1_score(y_test, pred, average='macro')
                    print(f"F1-macro: {f1:.3f}")
                    (_Path("reports/metrics_classification.txt")).write_text(f"F1-macro={f1:.4f}\n", encoding="utf-8")
            else:
                pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, pred)
                print(f"MAE: {mae:.3f}")
                (_Path("reports").mkdir(parents=True, exist_ok=True))
                (_Path("reports/metrics_regression.txt")).write_text(f"MAE={mae:.4f}\n", encoding="utf-8")
        elif skip_eval:
            print("[SKIP] Evaluación")
    except Exception as e:
        print(f"✖ entrenamiento/evaluación falló: {e}")
        ok = False

    # 3) XAI: Feature Importance y Permutation Importance
    try:
        if not skip_xai and model is not None:
            import numpy as _np
            import pandas as _pd
            from sklearn.inspection import permutation_importance
            rep_dir = _Path("reports"); rep_dir.mkdir(parents=True, exist_ok=True)
            # Feature importances (si aplica)
            if hasattr(model, "feature_importances_"):
                fi = _np.array(model.feature_importances_)
                feats = _pd.DataFrame({"feature": feature_names, "importance": fi})
                feats = feats.sort_values("importance", ascending=False)
                out = rep_dir / ("feature_importance_classification.csv" if task=="classification" else "feature_importance_regression.csv")
                feats.to_csv(out, index=False)
                print(f"✔ Feature importance guardado en {out}")
            # Permutation importance (submuestreo para velocidad)
            idx = _np.random.RandomState(42).choice(len(X_test), size=min(3000, len(X_test)), replace=False)
            perm = permutation_importance(model, X_test.iloc[idx], y_test.iloc[idx] if hasattr(y_test,'iloc') else y_test[idx], n_repeats=3, random_state=42, n_jobs=2)
            pi = _pd.DataFrame({
                "feature": feature_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False)
            out = rep_dir / ("permutation_importance_classification.csv" if task=="classification" else "permutation_importance_regression.csv")
            pi.to_csv(out, index=False)
            print(f"✔ Permutation importance guardado en {out}")
        elif skip_xai:
            print("[SKIP] XAI")
    except Exception as e:
        print(f"✖ XAI falló: {e}")
        ok = False

    list_artifacts()
    print("\n" + ("✅ Flujo completado" if ok else "⚠️ Flujo completado con errores"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
