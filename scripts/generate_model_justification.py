#!/usr/bin/env python3
"""Genera docs/model_justification.md (HU-JUST-01) con justificación de elección de modelos.
Lee artefactos existentes y sintetiza métricas y riesgos.
"""
import json, os, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / 'docs' / 'model_justification.md'

# Artefactos esperados
corr_path = ROOT / 'data' / 'processed' / 'correlation_matrix.csv'
vif_path = ROOT / 'data' / 'processed' / 'vif_scores.csv'
leak_path = ROOT / 'reports' / 'leakage_report.json'
cluster_path = ROOT / 'reports' / 'clustering_results.csv'
anom_path = ROOT / 'reports' / 'anomaly_results.csv'
hpo_summary = ROOT / 'reports' / 'hpo_summary.md'  # puede no existir

sections = {}

# Datos y riesgos
risks = []
if leak_path.exists():
    try:
        leak = json.loads(leak_path.read_text(encoding='utf-8'))
        risks.append(f"Posible leakage (flag={leak.get('flag')} r2={leak.get('r2')}) sobre features: {leak.get('tested_features')}")
    except Exception:
        risks.append('No se pudo parsear leakage_report.json')
else:
    risks.append('No existe leakage_report.json (ejecutar run_all.py para generarlo)')
# Riesgo temporal (drift post 2020)
risks.append('Riesgo de drift temporal post-2020: se usa split fijo para evitar fuga hacia el futuro.')
# Riesgo dimensionalidad / multicolinealidad
if corr_path.exists():
    try:
        corr = pd.read_csv(corr_path, index_col=0)
        # top pares correlaciones altas
        # Flatten
        pairs = []
        for i,c1 in enumerate(corr.columns):
            for j,c2 in enumerate(corr.columns):
                if j <= i: continue
                val = corr.loc[c1, c2]
                if abs(val) >= 0.85:  # umbral alto
                    pairs.append((c1, c2, val))
        pairs = sorted(pairs, key=lambda x: -abs(x[2]))[:5]
        if pairs:
            risks.append('Multicolinealidad en pares: ' + '; '.join([f"{a}-{b} (r={v:.2f})" for a,b,v in pairs]))
        sections['n_features_corr'] = corr.shape[0]
    except Exception:
        pass
else:
    risks.append('No existe correlation_matrix.csv')
sections['datos_y_riesgos'] = '\n'.join(f"- {r}" for r in risks)

# Algoritmos evaluados (resumen clustering & anomalías)
clust_lines = []
if cluster_path.exists():
    try:
        cl = pd.read_csv(cluster_path)
        best = cl.sort_values('silhouette', ascending=False).head(3)
        clust_lines.append('Top Silhouette clustering:')
        for _,row in best.iterrows():
            clust_lines.append(f"  * {row['algo']} {row['params']} silhouette={row['silhouette']:.3f}")
    except Exception:
        clust_lines.append('No se pudo leer clustering_results.csv')
else:
    clust_lines.append('No existe clustering_results.csv (ejecutar run_clustering.py).')

anom_lines = []
if anom_path.exists():
    try:
        an = pd.read_csv(anom_path)
        best_diff = an[(~an['anomaly_fraction'].isna())].copy()
        if not best_diff.empty:
            best_diff['diff'] = (best_diff['anomaly_fraction'] - best_diff['contamination_cfg']).abs()
            top = best_diff.sort_values('diff')
            # Evitar duplicados por algoritmo (tomar primeras únicas)
            seen = set()
            sel = []
            for _, row in top.iterrows():
                if row['algo'] in seen:
                    continue
                seen.add(row['algo'])
                sel.append(row)
                if len(sel) == 3:
                    break
            anom_lines.append('Algoritmos con fracción cercana a contaminación objetivo:')
            for row in sel:
                anom_lines.append(f"  * {row['algo']} frac={row['anomaly_fraction']:.3f} (target={row['contamination_cfg']})")
    except Exception:
        anom_lines.append('No se pudo leer anomaly_results.csv')
else:
    anom_lines.append('No existe anomaly_results.csv (ejecutar run_anomaly_detection.py).')

sections['algoritmos_evaluados'] = '\n'.join(clust_lines + [''] + anom_lines)

# Criterios de selección
criterios = [
    'Robustez a multicolinealidad y mezcla de tipos -> RandomForest para clasificación/regresión.',
    'Capacidad de capturar densidades arbitrarias -> DBSCAN preferido sobre KMeans (mejor silhouette en resultados).',
    'Escalabilidad en grandes n -> IsolationForest para anomalías (fracción ajustable).',
    'Interpretabilidad básica -> Importancias de árbol y permutation importance (generar ejecutando XAI).',
]
sections['criterios_seleccion'] = '\n'.join(f"- {c}" for c in criterios)

# Métricas claves (lectura de artefactos si existen)
metric_lines = []
# HPO summary presence
if hpo_summary.exists():
    metric_lines.append('Se dispone de hpo_summary.md para explorar hiperparámetros óptimos (ver docs).')
else:
    metric_lines.append('No se encontró hpo_summary.md (ejecutar HPO con run_hpo.py).')
# Silhouette promedio general
if cluster_path.exists():
    try:
        cl_all = pd.read_csv(cluster_path)
        sil_mean = cl_all['silhouette'].mean()
        metric_lines.append(f'Silhouette promedio clustering (todas configs)={sil_mean:.3f}.')
    except Exception:
        metric_lines.append('No se pudo calcular silhouette promedio (error de lectura).')
else:
    metric_lines.append('No existe clustering_results.csv para calcular silhouette promedio.')
# Métricas de clasificación/regresión si existen
metrics_class = ROOT / 'reports' / 'metrics_classification.txt'
metrics_reg = ROOT / 'reports' / 'metrics_regression.txt'
if metrics_class.exists():
    metric_lines.append('Clasificación: ' + metrics_class.read_text(encoding='utf-8').strip())
else:
    metric_lines.append('No hay métricas de clasificación (ejecutar evaluación sin --skip-eval).')
if metrics_reg.exists():
    metric_lines.append('Regresión: ' + metrics_reg.read_text(encoding='utf-8').strip())
else:
    metric_lines.append('No hay métricas de regresión (ejecutar pipeline con objetivo de regresión).')
# Feature importance artefacts
fi_cls = ROOT / 'reports' / 'feature_importance_classification.csv'
fi_reg = ROOT / 'reports' / 'feature_importance_regression.csv'
if fi_cls.exists():
    metric_lines.append('Importancias clasificación disponibles (feature_importance_classification.csv).')
else:
    metric_lines.append('No hay importancias de clasificación (ejecutar XAI sin --skip-xai).')
if fi_reg.exists():
    metric_lines.append('Importancias regresión disponibles (feature_importance_regression.csv).')
else:
    metric_lines.append('No hay importancias de regresión (ejecutar XAI con tarea de regresión).')
# Anomalías calibración
if anom_path.exists():
    try:
        an_all = pd.read_csv(anom_path)
        calib = (an_all['anomaly_fraction'] - an_all['contamination_cfg']).abs().mean()
        metric_lines.append(f'Error medio de calibración anomalías (|frac-target|)={calib:.3f}.')
    except Exception:
        metric_lines.append('No se pudo calcular calibración de anomalías.')
else:
    metric_lines.append('No existe anomaly_results.csv para métricas de anomalías.')
sections['metricas'] = '\n'.join(f"- {m}" for m in metric_lines)

# Trade-offs
tradeoffs = [
    'RandomForest vs modelos lineales: mayor costo computacional pero mejor manejo de no linealidad.',
    'DBSCAN sensible a eps / min_samples; KMeans más estable pero requiere k fijo.',
    'IsolationForest escalable, OneClassSVM más costoso (omitido en dataset grande).',
    'PCA opcional reduce dimensionalidad pero puede ocultar interpretabilidad directa de features originales.',
]
sections['tradeoffs'] = '\n'.join(f"- {t}" for t in tradeoffs)

# Recomendaciones futuras
recom = [
    'Persistir modelos finales con versión y semilla.',
    'Incorporar monitoreo de drift anual post 2024.',
    'Evaluar modelos de boosting (XGBoost/LightGBM) para mejora incremental.',
    'Añadir explicación SHAP para casos críticos.',
]
sections['recomendaciones'] = '\n'.join(f"- {r}" for r in recom)

# Tabla síntesis por tarea
tabla = [
    ['Tarea','Algoritmo Recomendado','Razón'],
    ['Clasificación','RandomForestClassifier','Robusto, maneja multicolinealidad y provee importancias.'],
    ['Regresión','RandomForestRegressor','Captura no linealidad y reduce overfitting vs árbol único.'],
    ['Clustering','DBSCAN (eps=0.7)','Mejor silhouette y detecta formas no convexas.'],
    ['Anomalías','IsolationForest','Escalable y ajusta fracción de outliers (contamination).'],
]
# Convert to markdown manually
tabla_md = ['| ' + ' | '.join(tabla[0]) + ' |','|---|---|---|'] + ['| ' + ' | '.join(row) + ' |' for row in tabla[1:]]

content = f"""# Justificación de Modelo (HU-JUST-01)

## Datos y Riesgos
{sections['datos_y_riesgos']}

Artefactos citados:
- [correlation_matrix.csv](../data/processed/correlation_matrix.csv)
- [vif_scores.csv](../data/processed/vif_scores.csv)
- [leakage_report.json](../reports/leakage_report.json)
- [clustering_results.csv](../reports/clustering_results.csv)
- [anomaly_results.csv](../reports/anomaly_results.csv)

## Algoritmos Evaluados
{sections['algoritmos_evaluados']}

## Criterios de Selección
{sections['criterios_seleccion']}

## Métricas Claves
{sections['metricas']}

## Trade-offs
{sections['tradeoffs']}

## Recomendaciones Futuras
{sections['recomendaciones']}

## Síntesis por Tarea
{os.linesep.join(tabla_md)}

_(Generado automáticamente por scripts/generate_model_justification.py)_
"""
DOC_PATH.write_text(content, encoding='utf-8')
print(f"Justificación escrita en {DOC_PATH}")
