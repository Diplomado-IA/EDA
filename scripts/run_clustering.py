#!/usr/bin/env python3
"""Script de clustering (HU-CLUST-01): KMeans y DBSCAN sobre X_train_engineered.
Genera metrics CSV y resumen Markdown con top configuraciones.
Uso:
  python scripts/run_clustering.py [--pca] [--save-plots] [--max-features 40] [--seed 42]
"""
import argparse, os, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
META_DIR = Path("outputs/metadata")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


def load_data(max_features: int | None = None) -> pd.DataFrame:
    x_path = DATA_DIR / "X_train_engineered.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"No existe {x_path}. Ejecuta Fase 3 primero.")
    X = pd.read_csv(x_path)
    # Reducir a primeras max_features si se especifica
    if max_features and X.shape[1] > max_features:
        X = X.iloc[:, :max_features]  # orden ya representa top seleccionadas
    # Filtrar variables no numéricas (should be numeric already)
    X = X.select_dtypes(include=[np.number]).copy()
    return X


def maybe_add_proxy_target(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    y_path = DATA_DIR / "y_train.csv"
    if y_path.exists():
        try:
            y_df = pd.read_csv(y_path)
            # Primer columna como serie
            y_series = y_df.iloc[:,0]
            if len(y_series) == len(X):
                return X, y_series
        except Exception:
            pass
    return X, None


def scale_and_reduce(X: pd.DataFrame, use_pca: bool, seed: int) -> tuple[pd.DataFrame, dict]:
    details = {}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    details["scaler_means"] = scaler.mean_.tolist()
    details["scaler_vars"] = scaler.var_.tolist()
    if use_pca and X.shape[1] > 50:
        pca = PCA(n_components=min(10, X.shape[1]), random_state=seed)
        Xp = pca.fit_transform(Xs)
        details["pca_explained_ratio"] = pca.explained_variance_ratio_.tolist()
        details["pca_n_components"] = pca.n_components_
        return pd.DataFrame(Xp), details
    return pd.DataFrame(Xs), details


def run_kmeans(X: pd.DataFrame, y, seed: int, sil_sample: int) -> list[dict]:
    results = []
    for k in [3,5,7,9]:
        t0 = time.perf_counter()
        print(f"[KMeans] k={k} iniciando...", flush=True)
        km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        labels = km.fit_predict(X)
        # Silhouette con submuestreo para evitar O(n^2) completo
        if len(np.unique(labels)) > 1:
            if sil_sample and len(X) > sil_sample:
                idx = np.random.default_rng(seed).choice(len(X), size=sil_sample, replace=False)
                sil = silhouette_score(X.iloc[idx], labels[idx])
            else:
                sil = silhouette_score(X, labels)
        else:
            sil = float('nan')
        hom = homogeneity_score(y, labels) if y is not None else math.nan
        comp = completeness_score(y, labels) if y is not None else math.nan
        runtime = (time.perf_counter() - t0) * 1000
        results.append({
            'algo': 'KMeans', 'params': {'n_clusters': k},
            'silhouette': sil, 'homogeneity': hom, 'completeness': comp,
            'runtime_ms': round(runtime,2)
        })
    return results


def run_dbscan(X: pd.DataFrame, y, seed: int, sil_sample: int, max_configs: int) -> list[dict]:
    results = []
    config_count = 0
    for eps in [0.3,0.5,0.7]:
        for ms in [5,10]:
            config_count += 1
            if max_configs and config_count > max_configs:
                print(f"[DBSCAN] Limite de configuraciones alcanzado ({max_configs}).")
                return results
            t0 = time.perf_counter()
            print(f"[DBSCAN] eps={eps} min_samples={ms} iniciando...", flush=True)
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            uniq = np.unique(labels)
            if len(uniq) > 1 and not (len(uniq)==1 and uniq[0]==-1):
                if sil_sample and len(X) > sil_sample:
                    idx = np.random.default_rng(seed).choice(len(X), size=sil_sample, replace=False)
                    sil = silhouette_score(X.iloc[idx], labels[idx])
                else:
                    sil = silhouette_score(X, labels)
            else:
                sil = float('nan')
            hom = homogeneity_score(y, labels) if y is not None and sil==sil else math.nan
            comp = completeness_score(y, labels) if y is not None and sil==sil else math.nan
            runtime = (time.perf_counter() - t0) * 1000
            results.append({
                'algo': 'DBSCAN', 'params': {'eps': eps, 'min_samples': ms},
                'silhouette': sil, 'homogeneity': hom, 'completeness': comp,
                'runtime_ms': round(runtime,2)
            })
    return results


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
        'script': 'run_clustering.py'
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    Path(META_DIR/'clustering_run.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


def save_results(results: list[dict]):
    df = pd.DataFrame(results)
    out_csv = REPORTS_DIR / 'clustering_results.csv'
    df.to_csv(out_csv, index=False)
    # Top 3 por silhouette
    top = df.sort_values('silhouette', ascending=False).head(3)
    # Usar CSV con separador '|' en lugar de markdown para evitar dependencia tabulate
    lines = ["# Resumen Clustering", "", "## Top 3 por Silhouette", top.to_csv(index=False, sep='|'), "", "Total configs: " + str(len(df))]
    (REPORTS_DIR/'clustering_summary.md').write_text("\n".join(lines), encoding='utf-8')
    return out_csv


def save_plots(results: list[dict]):
    try:
        import matplotlib.pyplot as plt; import seaborn as sns
    except Exception:
        return
    df = pd.DataFrame(results)
    # KMeans silhouette plot
    km = df[df.algo=='KMeans'].copy()
    if not km.empty:
        plt.figure(figsize=(6,4))
        sns.lineplot(data=km, x=km['params'].apply(lambda d:d['n_clusters']), y='silhouette', marker='o')
        plt.title('Silhouette vs n_clusters (KMeans)')
        plt.xlabel('n_clusters'); plt.ylabel('silhouette')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR/'clustering_kmeans_silhouette.png', dpi=120)
        plt.close()
    # DBSCAN heatmap eps/min_samples
    db = df[df.algo=='DBSCAN'].copy()
    if not db.empty:
        pivot = db.pivot_table(values='silhouette', index=db['params'].apply(lambda d:d['eps']), columns=db['params'].apply(lambda d:d['min_samples']))
        plt.figure(figsize=(6,4))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
        plt.title('Silhouette DBSCAN (eps vs min_samples)')
        plt.xlabel('min_samples'); plt.ylabel('eps')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR/'clustering_dbscan_eps_grid.png', dpi=120)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description='Clustering (KMeans/DBSCAN) rápido')
    ap.add_argument('--pca', action='store_true', help='Aplicar PCA si >50 columnas (n_components<=10)')
    ap.add_argument('--save-plots', action='store_true', help='Guardar gráficos de resultados')
    ap.add_argument('--max-features', type=int, default=None, help='Limitar a primeras N columnas')
    ap.add_argument('--seed', type=int, default=42, help='Semilla')
    ap.add_argument('--sil-sample', type=int, default=5000, help='Submuestreo para silhouette (0=sin submuestreo)')
    ap.add_argument('--sample-size', type=int, default=60000, help='Submuestreo filas antes de clustering (si dataset > sample-size)')
    ap.add_argument('--dbscan-max-configs', type=int, default=12, help='Limitar número de configuraciones DBSCAN (para acelerar)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    X = load_data(args.max_features)
    if args.sample_size and len(X) > args.sample_size:
        print(f"[INFO] Submuestreo filas: {args.sample_size} de {len(X)} para acelerar.")
        idx = np.random.default_rng(args.seed).choice(len(X), size=args.sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
    if X.shape[1] < 5:
        print(f"[WARN] Solo {X.shape[1]} columnas numéricas disponibles; silhouette puede ser poco informativa")
    Xs, details = scale_and_reduce(X, args.pca, args.seed)
    Xs, y = maybe_add_proxy_target(Xs)

    all_results = []
    all_results += run_kmeans(Xs, y, args.seed, args.sil_sample)
    all_results += run_dbscan(Xs, y, args.seed, args.sil_sample, args.dbscan_max_configs)

    out_csv = save_results(all_results)
    save_metadata(args.seed)
    if args.save_plots:
        save_plots(all_results)
        print('Gráficos guardados en reports/')
    print(f'Resultados clustering en {out_csv}')

if __name__ == '__main__':
    main()
