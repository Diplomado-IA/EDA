# Guía de Implementación EvM5U3
Timestamp: 2025-11-20T23:32:35.293Z

## Introducción
Este documento detalla las historias de usuario técnicas para completar los faltantes de la Actividad Sumativa N°3 (Unidad 3) relacionados con agrupamiento, detección de anomalías, justificación del modelo y presentación final. Cada historia incluye criterios de aceptación (formato Gherkin adaptado a ML) y tareas de programación para ejecución incremental.

---
## 1. Agrupamiento (Clustering)
### Historia de Usuario (HU-CLUST-01)
Como Analista de Datos quiero generar y comparar clusters sobre el dataset procesado para identificar segmentos homogéneos de programas que apoyen decisiones de oferta académica.

#### Contexto / Datos
- Entrada: `data/processed/X_train_engineered.csv` (sin columnas objetivo), opcional filtrado de top-N features seleccionadas.
- Algoritmos: KMeans y DBSCAN.

#### Criterios de Aceptación (Gherkin)
```
Given el dataset procesado sin targets
And existen al menos 5 variables numéricas
When ejecuto el script de clustering
Then se generan modelos KMeans y DBSCAN con parámetros por defecto y búsqueda rápida de hiperparámetros (n_clusters para KMeans, eps/min_samples para DBSCAN)
And se calcula Silhouette (>= -1 y <=1) para cada modelo válido
And se calculan homogeneidad y completitud si hay etiquetas proxy (e.g., MODALIDAD_BIN disponible)
And se genera un archivo reports/clustering_results.csv con filas por configuración y métricas
And se crea un resumen markdown reports/clustering_summary.md con Top 3 configuraciones por Silhouette
```
```
Given ejecuto clustering con bandera --save-plots
When finaliza el script
Then existen gráficos (reports/clustering_kmeans_silhouette.png, reports/clustering_dbscan_eps_grid.png) o se documenta ausencia por parámetros inválidos
```
```
Given el script corre sin errores
Then no modifica archivos fuera de reports/ y data/processed/
```

#### Tareas de Programación
- [ ] Crear `scripts/run_clustering.py`.
- [ ] Implementar carga de `X_train_engineered.csv`.
- [ ] Normalizar (si no ya escalado) y reducir dimensionalidad opcional (PCA rápido n_components=10 si >50 cols).
- [ ] Grid rápido KMeans (n_clusters=[3,5,7,9]).
- [ ] Grid rápido DBSCAN (eps in [0.3,0.5,0.7], min_samples in [5,10]).
- [ ] Cálculo métricas: silhouette (sklearn), homogeneidad/completitud si target clasificación disponible.
- [ ] Guardar CSV + Markdown resumen.
- [ ] Guardar gráficos silhouette vs n_clusters y heatmap eps/min_samples.
- [ ] Parametrizar vía argparse: --pca, --save-plots, --max-features, --seed.
- [ ] Actualizar README sección 8.x con “Clustering”.

---
## 2. Detección de Anomalías
### Historia de Usuario (HU-ANOM-01)
Como Ingeniero de ML quiero detectar observaciones atípicas en los datos para anticipar registros potencialmente erróneos o casos especiales.

#### Contexto / Datos
- Entrada: `data/processed/X_train_engineered.csv`
- Algoritmos: IsolationForest, LocalOutlierFactor (LOF).
- Proxy etiquetas: si se dispone de columna MODALIDAD_BIN en versión no filtrada del train (antes de drop objetivo) para calcular tasa de anomalías por clase.

#### Criterios de Aceptación (Gherkin)
```
Given el dataset de entrenamiento procesado
When ejecuto el script de anomalías
Then se ajustan modelos IsolationForest y LOF con parámetros default + grid rápido (contamination=[0.01,0.03,0.05])
And se calcula tasa de detección (proporción de puntos marcados anomalía) y FPR proxy si MODALIDAD_BIN existe
And se genera reports/anomalies_results.csv con columnas: algoritmo, params, contamination, detected_rate, proxy_fpr, runtime_ms
And se crea reports/anomalies_summary.md con explicación de interpretabilidad y tabla Top configuraciones por menor proxy_fpr o balance detection_rate
```
```
Given ejecuto el script con --score-export
Then se guarda reports/anomaly_scores.csv con puntuaciones crudas (score_samples) para IsolationForest
```
```
Given se genera un reporte
Then se incluyen recomendaciones (threshold sugerido) en anomalies_summary.md
```

#### Tareas de Programación
- [ ] Crear `scripts/run_anomalies.py`.
- [ ] Cargar X_train_engineered y (opcional) dataset con columna MODALIDAD_BIN para proxy.
- [ ] Implementar grid contamination.
- [ ] Calcular métricas básicas y tiempos (time.perf_counter()).
- [ ] Exportar CSV + Markdown resumen.
- [ ] Opción --score-export para guardar scores.
- [ ] Actualizar README sección 8.x con “Anomalías”.

---
## 3. Justificación de Modelo
### Historia de Usuario (HU-JUST-01)
Como Responsable Técnico quiero documentar la elección de algoritmos para cada tarea (clasificación, regresión, clustering, anomalías) sustentada en las características del dataset y métricas.

#### Criterios de Aceptación (Gherkin)
```
Given los artefactos de métricas y EDA existentes
When genero el documento de justificación
Then el archivo docs/model_justification.md incluye secciones: Datos y riesgos, Algoritmos evaluados, Criterios de selección, Métricas claves, Trade-offs, Recomendaciones futuras
And cita al menos 3 artefactos concretos (e.g., correlation_matrix.csv, feature_importance_regression.csv, hpo_summary.md)
And contiene una tabla síntesis por tarea con algoritmo recomendado y razón
```

#### Tareas de Programación
- [ ] Crear `docs/model_justification.md` plantilla inicial.
- [ ] Script opcional `scripts/generate_model_justification.py` que inserte métricas resumidas.
- [ ] Añadir links relativos a artefactos en el markdown.
- [ ] Referenciar riesgos (data_leakage, drift_post_2020, etc.).
- [ ] Actualizar README para enlazar el documento.

---
## 4. Presentación Final
### Historia de Usuario (HU-PRESENT-01)
Como Facilitador necesito una presentación ejecutiva que sintetice problema, datos, modelos, resultados y próximos pasos.

#### Criterios de Aceptación (Gherkin)
```
Given los artefactos clave generados
When ejecuto el generador de presentación
Then se crea docs/presentacion.md con estructura: Contexto, Dataset, Metodología, Resultados (clasificación, regresión, clustering, anomalías), Interpretabilidad, Recomendaciones
And cada sección incluye al menos una métrica o figura referenciada
And opcionalmente se exporta a PDF (si librería disponible) sin errores
```
```
Given ejecuto con --export-pptx (si dependencias instaladas)
Then se genera docs/presentacion.pptx con 8–12 diapositivas y títulos consistentes
```

#### Tareas de Programación
- [ ] Crear `scripts/generate_presentation.py` con plantillas markdown.
- [ ] Insertar tablas y métricas recientes (leer reports/*, outputs/hpo_*). 
- [ ] Opción --export-pdf (usar pypandoc/reportlab si disponible) y --export-pptx (python-pptx).
- [ ] Validar existencia de artefactos antes de incluirlos (fallback: mensaje “pendiente”).
- [ ] Actualizar README con instrucción de uso.

---
## 5. Consideraciones Transversales
### Historia (HU-X-ROBUST-01)
Como Equipo de Calidad quiero garantizar trazabilidad y reproducibilidad mínima.
```
Given ejecuto cualquier script nuevo (clustering, anomalías)
Then agrega un bloque metadata JSON (timestamp UTC, git_sha, random_state) en outputs/metadata/*.json
And no expone datos sensibles
```

### Tareas Transversales
- [ ] Utilizar helper común para metadata (crear `src/utils/metadata.py`).
- [ ] Reutilizar semilla global Config.RANDOM_STATE.
- [ ] Validar tamaños de salida y registrar runtime_ms.

---
## 6. Prioridad y Orden Recomendada
1. Clustering (HU-CLUST-01)
2. Anomalías (HU-ANOM-01)
3. Justificación (HU-JUST-01)
4. Presentación (HU-PRESENT-01)
5. Transversal (HU-X-ROBUST-01)

---
## 7. Métricas de Éxito Internas
- Cobertura de historias implementadas ≥80%
- Tiempo de ejecución clustering/anomalías (fast mode) < 60s en dataset actual
- Documento de justificación ≥ 600 palabras
- Presentación generada sin errores y con ≥8 secciones

---
## 8. Riesgos y Mitigaciones
| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Altas dimensiones ralentizan clustering | Silhouette costoso | PCA opcional y fast grid |
| Ausencia de etiqueta proxy para anomalías | Métrica FPR no calculable | Documentar limitación y usar contamination descriptivo |
| Falta de dependencias para exportar PDF/PPTX | No se genera presentación final | Incluir flags y fallback a sólo markdown |
| Data leakage reintroducido en scripts nuevos | Métricas infladas | Reutilizar versiones sin target en clustering/anomalías |

---
## 9. Checklist de Cierre
- [ ] scripts/run_clustering.py creado y probado
- [ ] scripts/run_anomalies.py creado y probado
- [ ] reports/clustering_results.csv y clustering_summary.md
- [ ] reports/anomalies_results.csv y anomalies_summary.md
- [ ] docs/model_justification.md completo
- [ ] docs/presentacion.md generado
- [ ] README actualizado con nuevas secciones
- [ ] Metadata JSON en outputs/metadata para nuevos scripts

---
## 10. Mantenimiento Futuro
- Extender clustering a métodos adicionales (Agglomerative, HDBSCAN) si se requiere granularidad.
- Añadir umbrales adaptativos para anomalías (percentiles dinámicos).
- Integrar panel interactivo de clustering/anomalías en la UI (visualización dinámica de Silhouette y scores).
- Incorporar pruebas unitarias básicas en `tests/` para garantizar regresión mínima.

Fin del documento.
