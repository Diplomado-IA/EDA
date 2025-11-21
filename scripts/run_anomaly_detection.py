#!/usr/bin/env python3
"""Script de detección de anomalías (HU-ANOM-01).
Aplica varios algoritmos no supervisados sobre X_train_engineered para identificar registros atípicos.
Genera:
  - reports/anomaly_results.csv (resumen por algoritmo)
  - reports/anomaly_summary.md (Top algoritmos por fracción de anomalías dentro del rango esperado)
  - reports/anomaly_scores_*.png (distribuciones de scores si --save-plots)
Uso:
  python scripts/run_anomaly_detection.py [--pca] [--save-plots] [--max-features N] [--seed S]
                                        [--sample-size M] [--contamination C]
Parámetros:
  --contamination C  Proporción esperada de anomalías (p.ej. 0.05) para métodos que lo soportan.
Notas:
  - Requiere haber ejecutado previamente el preprocesamiento que genera data/processed/X_train_engineered.csv
  - No depende de tabulate (usa to_csv con separador '|').
"""
import argparse, os, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
# OneClassSVM es más costoso; se deja opcional.
from sklearn.svm import OneClassSVM

DATA_DIR = Path('data/processed')
REPORTS_DIR = Path('reports'); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
META_DIR = Path('outputs/metadata'); META_DIR.mkdir(parents=True, exist_ok=True)


def load_data(max_features: int | None = None) -> pd.DataFrame:
    x_path = DATA_DIR / 'X_train_engineered.csv'
    if not x_path.exists():
        raise FileNotFoundError(f'No existe {x_path}. Ejecuta primero: python scripts/run_all.py --skip-eval --skip-xai')
    X = pd.read_csv(x_path)
    if max_features and X.shape[1] > max_features:
        X = X.iloc[:, :max_features]
    X = X.select_dtypes(include=[np.number]).copy()
    return X


def scale_and_reduce(X: pd.DataFrame, use_pca: bool, seed: int) -> pd.DataFrame:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if use_pca and X.shape[1] > 50:
        pca = PCA(n_components=min(10, X.shape[1]), random_state=seed)
        Xs = pca.fit_transform(Xs)
    return pd.DataFrame(Xs)


def run_algorithms(X: pd.DataFrame, seed: int, contamination: float, sample_size: int | None) -> list[dict]:
    results = []
    n = len(X)
    # Submuestreo previo si se pide
    if sample_size and n > sample_size:
        print(f'[INFO] Submuestreo filas: {sample_size} de {n} para acelerar.')
        idx = np.random.default_rng(seed).choice(n, size=sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        n = len(X)
    # Definición de configuraciones
    algos = []
    algos.append(('IsolationForest', IsolationForest(random_state=seed, contamination=contamination, n_jobs=-1, n_estimators=200), {}))
    # LocalOutlierFactor: dos configuraciones
    for neighbors in [20, 40]:
        algos.append(('LocalOutlierFactor', LocalOutlierFactor(n_neighbors=neighbors, contamination=contamination, novelty=False), {'n_neighbors': neighbors}))
    # EllipticEnvelope (asume distribución ~gaussiana)
    algos.append(('EllipticEnvelope', EllipticEnvelope(contamination=contamination, random_state=seed), {}))
    # OneClassSVM (costoso); usar tamaño reducido si > 25000
    if n <= 25000:
        algos.append(('OneClassSVM', OneClassSVM(kernel='rbf', gamma='scale', nu=contamination), {}))
    else:
        print('[SKIP] OneClassSVM (dataset grande, >25k registros)')
    for name, model, extra in algos:
        t0 = time.perf_counter()
        print(f'[{name}] iniciando...', flush=True)
        labels = None
        scores = None
        try:
            if name == 'LocalOutlierFactor' and not model.novelty:
                labels = model.fit_predict(X)
                # score_samples no disponible sin novelty; usar negative_outlier_factor_
                scores = model.negative_outlier_factor_
            else:
                model.fit(X)
                labels = model.predict(X)  # IsolationForest: 1 normal, -1 outlier; OneClassSVM/Elliptic igual
                if hasattr(model, 'score_samples'):
                    scores = model.score_samples(X)
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
            runtime = (time.perf_counter() - t0) * 1000
            # Calcular fracción de anomalías
            frac = float(np.mean(labels == -1)) if labels is not None else math.nan
            # Simple estadística de scores
            score_mean = float(np.nanmean(scores)) if scores is not None else math.nan
            score_std = float(np.nanstd(scores)) if scores is not None else math.nan
            results.append({
                'algo': name,
                'params': json.dumps(extra, ensure_ascii=False),
                'contamination_cfg': contamination,
                'anomaly_fraction': frac,
                'score_mean': score_mean,
                'score_std': score_std,
                'n_samples': n,
                'runtime_ms': round(runtime, 2)
            })
            # Guardar scores individuales por algoritmo (CSV)
            try:
                if scores is not None:
                    out_scores = REPORTS_DIR / f'anomaly_scores_{name}.csv'
                    pd.DataFrame({'score': scores, 'label': labels}).to_csv(out_scores, index=False)
            except Exception:
                pass
        except Exception as e:
            runtime = (time.perf_counter() - t0) * 1000
            results.append({
                'algo': name,
                'params': json.dumps(extra, ensure_ascii=False),
                'contamination_cfg': contamination,
                'anomaly_fraction': math.nan,
                'score_mean': math.nan,
                'score_std': math.nan,
                'n_samples': n,
                'runtime_ms': round(runtime, 2),
                'error': str(e)
            })
            print(f'[WARN] {name} falló: {e}')
    return results


def save_results(results: list[dict]):
    df = pd.DataFrame(results)
    out_csv = REPORTS_DIR / 'anomaly_results.csv'
    df.to_csv(out_csv, index=False)
    # Top algoritmos cuya fracción se acerca a contamination (orden por |anomaly_fraction - contamination| asc)
    if 'anomaly_fraction' in df.columns:
        df_ok = df[df['anomaly_fraction'].notna()].copy()
        df_ok['diff_target'] = (df_ok['anomaly_fraction'] - df_ok['contamination_cfg']).abs()
        top = df_ok.sort_values('diff_target').head(3)
    else:
        top = df.head(3)
    lines = [
        '# Resumen Anomalías',
        '',
        '## Top 3 por cercanía a contaminación esperada',
        top.to_csv(index=False, sep='|'),
        '',
        f'Total algoritmos: {len(df)}'
    ]
    (REPORTS_DIR / 'anomaly_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return out_csv


def save_plots(results: list[dict]):
    try:
        import matplotlib.pyplot as plt; import seaborn as sns
    except Exception:
        return
    df = pd.DataFrame(results)
    if not df.empty:
        plt.figure(figsize=(6,4))
        sns.barplot(data=df, x='algo', y='anomaly_fraction')
        plt.title('Fracción de anomalías por algoritmo')
        plt.ylabel('anomaly_fraction')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'anomaly_fraction_by_algo.png', dpi=120)
        plt.close()
    # Scores distributions (si existen archivos generados)
    for name in df['algo'].unique():
        score_file = REPORTS_DIR / f'anomaly_scores_{name}.csv'
        if score_file.exists():
            try:
                sc_df = pd.read_csv(score_file)
                plt.figure(figsize=(6,4))
                sns.histplot(sc_df['score'], bins=40, kde=True)
                plt.title(f'Distribución scores {name}')
                plt.tight_layout()
                plt.savefig(REPORTS_DIR / f'anomaly_scores_dist_{name}.png', dpi=120)
                plt.close()
            except Exception:
                pass


def save_metadata(seed: int):
    sha = None
    try:
        import subprocess
        sha = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
    except Exception:
        pass
    meta = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_sha': sha,
        'seed': seed,
        'script': 'run_anomaly_detection.py'
    }
    Path(META_DIR / 'anomaly_run.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Detección de anomalías rápida (IsolationForest, LOF, EllipticEnvelope, OneClassSVM opcional)')
    ap.add_argument('--pca', action='store_true', help='Aplicar PCA si >50 columnas')
    ap.add_argument('--save-plots', action='store_true', help='Guardar gráficos de resultados y distribuciones')
    ap.add_argument('--max-features', type=int, default=None, help='Limitar a primeras N columnas')
    ap.add_argument('--seed', type=int, default=42, help='Semilla')
    ap.add_argument('--sample-size', type=int, default=60000, help='Submuestreo filas antes de algoritmos (si dataset > sample-size)')
    ap.add_argument('--contamination', type=float, default=0.05, help='Proporción esperada de anomalías')
    args = ap.parse_args()

    np.random.seed(args.seed)
    X = load_data(args.max_features)
    Xs = scale_and_reduce(X, args.pca, args.seed)

    results = run_algorithms(Xs, args.seed, args.contamination, args.sample_size)
    out_csv = save_results(results)
    save_metadata(args.seed)
    if args.save_plots:
        save_plots(results)
        print('Gráficos guardados en reports/')
    print(f'Resultados anomalías en {out_csv}')

if __name__ == '__main__':
    main()
