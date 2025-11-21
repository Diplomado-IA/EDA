# Justificación de Modelo (HU-JUST-01)

## Datos y Riesgos
- Posible leakage (flag=False r2=0.18854441022979995) sobre features: ['PROMEDIO EDAD HOMBRE ', 'PROMEDIO EDAD MUJER ']
- Riesgo de drift temporal post-2020: se usa split fijo para evitar fuga hacia el futuro.
- Multicolinealidad en pares: TOTAL TITULACIONES-TOTAL RANGO EDAD (r=1.00); CARRERA CLASIFICACIÓN NIVEL 2_Carreras Técnicas-CARRERA CLASIFICACIÓN NIVEL 1_Técnico de Nivel Superior (r=1.00); CLASIFICACIÓN INSTITUCIÓN NIVEL 3_Universidades Privadas-CLASIFICACIÓN INSTITUCIÓN NIVEL 2_Universidades Privadas (r=1.00); CLASIFICACIÓN INSTITUCIÓN NIVEL 1_Institutos Profesionales-CLASIFICACIÓN INSTITUCIÓN NIVEL 3_Institutos Profesionales (r=1.00); CLASIFICACIÓN INSTITUCIÓN NIVEL 1_Institutos Profesionales-CLASIFICACIÓN INSTITUCIÓN NIVEL 2_Institutos Profesionales (r=1.00)

Artefactos citados:
- [correlation_matrix.csv](../data/processed/correlation_matrix.csv)
- [vif_scores.csv](../data/processed/vif_scores.csv)
- [leakage_report.json](../reports/leakage_report.json)
- [clustering_results.csv](../reports/clustering_results.csv)
- [anomaly_results.csv](../reports/anomaly_results.csv)

## Algoritmos Evaluados
Top Silhouette clustering:
  * DBSCAN {'eps': 0.7, 'min_samples': 5} silhouette=0.219
  * KMeans {'n_clusters': 7} silhouette=0.182
  * DBSCAN {'eps': 0.7, 'min_samples': 10} silhouette=0.182

Algoritmos con fracción cercana a contaminación objetivo:
  * IsolationForest frac=0.050 (target=0.05)
  * LocalOutlierFactor frac=0.050 (target=0.05)
  * EllipticEnvelope frac=0.050 (target=0.05)

## Criterios de Selección
- Robustez a multicolinealidad y mezcla de tipos -> RandomForest para clasificación/regresión.
- Capacidad de capturar densidades arbitrarias -> DBSCAN preferido sobre KMeans (mejor silhouette en resultados).
- Escalabilidad en grandes n -> IsolationForest para anomalías (fracción ajustable).
- Interpretabilidad básica -> Importancias de árbol y permutation importance (generar ejecutando XAI).

## Métricas Claves
- Se dispone de hpo_summary.md para explorar hiperparámetros óptimos (ver docs).
- Silhouette promedio clustering (todas configs)=0.088.
- Clasificación: AUC-ROC=0.4644
- No hay métricas de regresión (ejecutar pipeline con objetivo de regresión).
- Importancias clasificación disponibles (feature_importance_classification.csv).
- No hay importancias de regresión (ejecutar XAI con tarea de regresión).
- Error medio de calibración anomalías (|frac-target|)=0.000.

## Trade-offs
- RandomForest vs modelos lineales: mayor costo computacional pero mejor manejo de no linealidad.
- DBSCAN sensible a eps / min_samples; KMeans más estable pero requiere k fijo.
- IsolationForest escalable, OneClassSVM más costoso (omitido en dataset grande).
- PCA opcional reduce dimensionalidad pero puede ocultar interpretabilidad directa de features originales.

## Recomendaciones Futuras
- Persistir modelos finales con versión y semilla.
- Incorporar monitoreo de drift anual post 2024.
- Evaluar modelos de boosting (XGBoost/LightGBM) para mejora incremental.
- Añadir explicación SHAP para casos críticos.

## Síntesis por Tarea
| Tarea | Algoritmo Recomendado | Razón |
|---|---|---|
| Clasificación | RandomForestClassifier | Robusto, maneja multicolinealidad y provee importancias. |
| Regresión | RandomForestRegressor | Captura no linealidad y reduce overfitting vs árbol único. |
| Clustering | DBSCAN (eps=0.7) | Mejor silhouette y detecta formas no convexas. |
| Anomalías | IsolationForest | Escalable y ajusta fracción de outliers (contamination). |

_(Generado automáticamente por scripts/generate_model_justification.py)_
