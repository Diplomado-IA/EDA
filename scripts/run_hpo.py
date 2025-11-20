#!/usr/bin/env python3
"""
HPO para RandomForest (regresión o clasificación) soportando GridSearchCV y BayesSearchCV.
Genera: results.csv, best.json y reports/hpo_summary.md con tabla comparativa.
"""
import argparse
import os
import sys
import json
import subprocess
from datetime import datetime

import pandas as pd

# Importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.splits import get_cv
from src.models.train import run_grid_search
from src.features.leakage import detect_leakage, save_leakage_report


def read_csv_robust(path: str, encoding: str | None = None, sep: str | None = None) -> pd.DataFrame:
    encodings = [encoding] if encoding else ["utf-8", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep if sep is not None else None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            if encoding is None:
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("No se pudo leer el CSV con las codificaciones probadas")


def load_config(path: str = "config/params.yaml") -> dict:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def prepare_data(args, cfg, task: str):
    if args.data:
        df = read_csv_robust(args.data, encoding=args.encoding, sep=args.sep)
        df.columns = [str(c).strip() for c in df.columns]
        assert args.target, "Debe especificar --target cuando se usa --data"
        target = args.target
        if cfg.get("cv", {}).get("kind") == "time" and args.date_col:
            if args.date_col not in df.columns:
                raise ValueError(f"date-col {args.date_col} no existe en datos")
            df = df.sort_values(args.date_col)
        y = df[target]
        X = df.drop(columns=[target])
        X = X.select_dtypes(include=['number']).copy().fillna(X.median(numeric_only=True))
        return X, y, df
    if task == "clf":
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer()
        X = pd.DataFrame(ds.data, columns=ds.feature_names)
        y = pd.Series(ds.target, name="target")
        return X, y, None
    from sklearn.datasets import fetch_california_housing
    ds = fetch_california_housing()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, None


def maybe_check_leakage(df, target, cfg):
    if df is None or not cfg.get("leakage", {}).get("detect", True):
        return None
    suspects = cfg.get("leakage", {}).get("suspect_features", [])
    r2_thr = cfg.get("leakage", {}).get("r2_threshold", 0.9)
    rep = detect_leakage(df, target, suspects, r2_thr)
    save_leakage_report(rep)
    return rep


def build_estimator(cfg, task: str, n_jobs: int = -1, max_samples: int | None = None):
    rs = cfg.get("hpo", {}).get("random_state", 42)
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if task == "clf":
        return RandomForestClassifier(random_state=rs, n_estimators=200, n_jobs=n_jobs, max_samples=max_samples)
    return RandomForestRegressor(random_state=rs, n_estimators=200, n_jobs=n_jobs, max_samples=max_samples)


def build_param_grid(cfg, fast: bool):
    grid = cfg.get("hpo", {}).get("grid", {})
    if fast:
        return {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
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
    meta = {"timestamp": datetime.utcnow().isoformat() + "Z", "git_sha": sha}
    md_dir = os.path.join("outputs", "metadata")
    os.makedirs(md_dir, exist_ok=True)
    with open(os.path.join(md_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_summary(out_dir: str, scoring_list, refit_metric: str):
    import pandas as _pd
    rep_dir = "reports"; os.makedirs(rep_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.csv")
    if not os.path.exists(res_path):
        return
    df = _pd.read_csv(res_path)
    param_cols = [c for c in df.columns if c.startswith("param_")]
    lines = ["# Resumen HPO", f"Refit metric: {refit_metric}", ""]
    for metric in scoring_list:
        col = f"mean_test_{metric}"
        if col not in df.columns:
            continue
        top = df.sort_values(col, ascending=False).head(5)[param_cols + [col]]
        lines.append(f"## Top 5 por {metric}")
        lines.append("|" + "|".join(top.columns) + "|")
        lines.append("|" + "|".join(["---"] * len(top.columns)) + "|")
        for _, r in top.iterrows():
            lines.append("|" + "|".join(str(r[c]) for c in top.columns) + "|")
        lines.append("")
    best_json = os.path.join(out_dir, "best.json")
    if os.path.exists(best_json):
        bj = json.loads(open(best_json, "r", encoding="utf-8").read())
        lines.append("## Mejor configuración")
        lines.append("```json")
        lines.append(json.dumps(bj, ensure_ascii=False, indent=2))
        lines.append("```")
    summary_path = os.path.join(rep_dir, "hpo_summary.md")
    mode = "a" if os.path.exists(summary_path) else "w"
    # Delimitador entre ejecuciones
    if mode == "a":
        lines.insert(0, "\n---\n")
    with open(summary_path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Ejecuta HPO (Grid/Bayes) para RandomForest")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--sep", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/hpo")
    parser.add_argument("--no-leak-check", action="store_true")
    parser.add_argument("--task", type=str, choices=["reg", "clf"], default=None)
    parser.add_argument("--method", type=str, choices=["grid", "bayes"], default=None)
    parser.add_argument("--bayes-iter", type=int, default=25)
    parser.add_argument("--fast", action="store_true", help="Reduce grid y n_iter para ejecución rápida")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Núcleos para entrenamiento de RandomForest")
    parser.add_argument("--max-samples", type=int, default=None, help="Submuestreo bootstrap (RandomForest)")
    args = parser.parse_args()

    cfg = load_config()
    task = args.task or cfg.get("hpo", {}).get("task", "reg")
    task = "clf" if task.startswith("clf") or task.startswith("class") else "reg"
    method = (args.method or cfg.get("hpo", {}).get("method", "grid")).lower()
    fast = args.fast

    X, y, df_raw = prepare_data(args, cfg, task)

    if task == "reg" and (not args.no_leak_check) and df_raw is not None and args.target:
        maybe_check_leakage(df_raw, args.target, cfg)

    cv = get_cv(cfg.get("cv", {}))
    # Ajuste automático de max_samples si excede n_samples totales
    max_samples_eff = args.max_samples
    if max_samples_eff is not None and max_samples_eff > len(X):
        print(f"[INFO] max_samples={max_samples_eff} > n_samples={len(X)} -> usando max_samples={len(X)}")
        max_samples_eff = len(X)
    estimator = build_estimator(cfg, task, n_jobs=args.n_jobs, max_samples=max_samples_eff)
    param_grid = build_param_grid(cfg, fast)

    if task == "reg":
        scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
        refit_metric = "neg_root_mean_squared_error"
    else:
        scoring = ["roc_auc", "f1_macro", "accuracy"]
        refit_metric = "roc_auc"

    os.makedirs(args.out_dir, exist_ok=True)

    if method == "grid":
        gs = run_grid_search(estimator, param_grid, X, y, cv, scoring=scoring, out_dir=args.out_dir)
    else:
        try:
            from skopt import BayesSearchCV
            from skopt.space import Integer, Categorical
            search_spaces = {}
            for p, vals in param_grid.items():
                vals_clean = [v for v in vals]
                if any(v is None for v in vals_clean) or any(not isinstance(v, (int, float)) for v in vals_clean):
                    search_spaces[p] = Categorical(vals_clean)
                else:
                    search_spaces[p] = Integer(int(min(vals_clean)), int(max(vals_clean)))
            bayes = BayesSearchCV(
                estimator=estimator,
                search_spaces=search_spaces,
                n_iter=(5 if fast else args.bayes_iter),
                cv=cv,
                scoring=scoring,
                refit=refit_metric,
                n_jobs=-1,
                random_state=cfg.get("hpo", {}).get("random_state", 42),
                return_train_score=True,
            )
            bayes.fit(X, y)
            pd.DataFrame(bayes.cv_results_).to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
            best = {"best_params_": bayes.best_params_, "best_score_": bayes.best_score_, "refit_metric": refit_metric, "method": "bayes"}
            with open(os.path.join(args.out_dir, "best.json"), "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] BayesSearchCV no disponible ({e}). Fallback a grid.")
            run_grid_search(estimator, param_grid, X, y, cv, scoring=scoring, out_dir=args.out_dir)

    persist_run_metadata(args.out_dir)
    write_summary(args.out_dir, scoring, refit_metric)
    print(f"HPO finalizado. Método={method} Tarea={task}. Resultados en {args.out_dir}")


if __name__ == "__main__":
    main()
