
# 🚀 Proyecto ML – Arquitectura Modular con UI en Streamlit

Este repositorio contiene un flujo completo de **EDA → Preprocesamiento → Artefactos ML**, expuesto a través de una **UI interactiva en Streamlit** y estructurado según una **arquitectura modular**.


## 📦 1) Descarga del proyecto

### Requisitos previos

- **Git**
- **Python 3.10+**

### Clonar el repositorio

```bash
git clone <URL_DEL_REPO>
cd EDA
```

> 💡 Asegúrate de estar en la carpeta raíz del proyecto antes de continuar.


## 🛠️ 2) Configuración básica

### Crear y activar entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
# En Windows:
# venv\Scripts\activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```


## 📂 3) Dataset y configuración

### Ubicación del dataset

Verifica que el archivo CSV esté en:

```text
data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv
```

### Configuración actual

El archivo de configuración principal es:

```text
config/config.py
```

Allí se definen, entre otros:

* Separador del CSV: `';'`
* *Encoding*: `'latin1'`

> ⚙️ Si cambias el archivo de entrada o su formato, **ajusta estos parámetros** en `config/config.py`.

### Objetivos del modelo

* **Clasificación (`MODALIDAD_BIN`)**

  * Clase `1` → **Presencial**
  * Clase `0` → **No presencial / otras modalidades**

* **Regresión (`PROMEDIO_EDAD_PROGRAMA`)**

  * Variable continua de edad promedio por programa.



## 🎛️ 4) Ejecutar la UI (Streamlit)

### Lanzar la aplicación

```bash
streamlit run ui/app.py
```

### Secciones disponibles en la UI

* **Fase 1 – Configuración inicial**

  * Validar objetivos (`MODALIDAD_BIN`, `PROMEDIO_EDAD_PROGRAMA`).
  * Verificar ruta y parámetros de lectura del dataset.

* **Fase 2 – EDA (Análisis Exploratorio de Datos)**

  * Carga del dataset.
  * Ejecución del EDA automatizado.
  * Visualización de artefactos generados (`.csv`, `.png`) con su ruta correspondiente.

* **Fase 3 – Preprocesamiento**

  * Limpieza de datos.
  * *Split* temporal.
  * Escalado con **StandardScaler**.
  * Codificación segura de variables categóricas:

    * **One-Hot Encoding (OHE)** con *rare grouping* / *frequency encoding*.
  * Generación y cálculo de *features*:

    * **HHI**
    * **LQ**
    * **IPG**
  * Cálculo optimizado de:

    * **Matriz de correlación**
    * **VIF (Variance Inflation Factor)**
  * Selección de variables y guardado de resultados.

* **Fase 4 – Interpretabilidad (XAI)**

  * Entrena un modelo demo (RandomForest/Logistic/Linear) sobre train.
  * Explicabilidad: Feature Importance (árbol), Permutation Importance y Coeficientes lineales.
  * Guarda artefactos en `reports/*.csv` y muestra tablas/gráficos en la UI.

* **Informes**

  * Pestañas que renderizan todos los `.md` dentro de `docs/`.

* **Botón lateral**

  * **"Limpiar artefactos (clean.sh)"**
    Permite reiniciar la salida del proyecto sin modificar los datos crudos en `data/raw`.


## 📁 5) Artefactos generados

### EDA / Resúmenes

* `outputs/eda/resumen/*`
  Incluye:

  * CSVs de resumen
  * `decision_metricas.txt`

### Gráficos

* `outputs/eda/figures/*`
* Copias auxiliares en:

  * `data/processed/*.png`

### Correlación / VIF

* `data/processed/correlation_matrix.csv`
* `data/processed/vif_scores.csv`
* Archivos auxiliares:

  * `*columns_used.txt` (columnas empleadas para los cálculos)

### Selección de *features*

* `data/processed/selected_features.txt`

### Datasets finales

* `data/processed/X_train_engineered.csv`
* `data/processed/X_test_engineered.csv`

### Interpretabilidad (XAI)

* `reports/feature_importance_*.csv`
* `reports/permutation_importance_*.csv`
* `reports/coefficients_linear_*.csv`




## 🧪 6) Ejecución desde CLI (flujo completo)

Si prefieres correr el flujo sin UI:

### 6.1 Activar entorno e instalar dependencias

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Ejecutar flujo completo (EDA + preprocesamiento)

```bash
python scripts/run_all.py
```

### 6.3 Artefactos generados vía CLI

* `data/processed/*`

  * Datasets procesados
  * Correlación
  * VIF
  * *Features* seleccionadas

* `outputs/eda/resumen/*`

  * Resúmenes de EDA y preprocesamiento


## 🧾 7) Notas y convenciones

* **ML** → *Machine Learning* (Aprendizaje Automático)
* **OHE** → *One-Hot Encoding*
* **VIF** → *Variance Inflation Factor*

Si cambias los objetivos (`MODALIDAD_BIN` / `PROMEDIO_EDAD_PROGRAMA`), recuerda actualizar:

* `config/config.py`

### Script de limpieza: `clean.sh`

```bash
bash clean.sh
```

* Recrea la estructura de artefactos **vacía**.
* **No modifica** el contenido de `data/raw`.

### Limitar uso de CPU en cálculos intensivos (opcional)

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```


## 8) Implementación de oportunidad de mejora

### 8.1 Riesgo de fuga/tautología
**Oportunidad**: Riesgo de fuga/tautología al predecir `PROMEDIO EDAD PROGRAMA ` usando sus componentes directos (`PROMEDIO EDAD HOMBRE `, `PROMEDIO EDAD MUJER `), lo que podría inflar métricas sin aportar señal nueva.
Implementación: Se añadió detección automática (regresión lineal simple) en `preprocess_pipeline` con limpieza robusta (strip, normalización decimal) y reporte `reports/leakage_report.json`. Estrategias soportadas en `config/params.yaml`: `drop_features`, `redefine_target`, `fail`. Umbral `r2_threshold=0.90` mantiene criterios estrictos; en los datos actuales R²≈0.19 < 0.90 ⇒ no se aplica mitigación.
Cómo ejecutar/prueba:
1. Flujo completo: `python scripts/run_all.py` (muestra resumen [LEAKAGE] en consola y genera `reports/leakage_report.json`).
2. Chequeo directo regresión: `python scripts/run_regression_leakage.py --strategy redefine_target` (fuerza lectura y pipeline; si R²≥0.90 redefine y crea `outputs/metadata/target_mapping.json`).
3. Para probar mitigaciones: cambia `strategy` a `drop_features` o `redefine_target` y (opcional) ajusta temporalmente `r2_threshold` a un valor menor (ej. 0.05) para ver acción aplicada (`reports/leakage_action.txt`).
Timestamp actualización: 2025-11-20T01:56:36.110Z
 
---

### 8.2 Tuning explícito (HPO)
**Oportunidad**: originalmente el entrenamiento usaba hiperparámetros fijos en RandomForest, sin experimento sistemático ni documentación de la selección óptima. Se requería explorar `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` con métodos Grid y Bayes y dejar trazabilidad.
Implementación: se creó `scripts/run_hpo.py` con soporte para:
- GridSearchCV (`--method grid`) usando el espacio configurado en `config/params.yaml`.
- BayesSearchCV (`--method bayes`) para optimización más eficiente.
- Tareas de regresión y clasificación (`--task reg|clf`).
- Modo rápido (`--fast`) que reduce el grid / iteraciones para exploración inicial.
- Submuestreo opcional `--max-samples` y control de núcleos `--n-jobs`.
Artefactos generados:
- `outputs/hpo_<task>/results.csv` con todas las combinaciones y métricas (`neg_mean_absolute_error`, `neg_root_mean_squared_error`, `r2` o `roc_auc`, `f1_macro`, `accuracy`).
- `outputs/hpo_<task>/best.json` con la mejor configuración según métrica de refit.
- `reports/hpo_summary.md` acumulando ejecuciones y mostrando Top 5 por métrica y bloque JSON de la mejor configuración.
Integración en flujo principal: bandera `--with-hpo` en `scripts/run_all.py` dispara HPO previo al entrenamiento y aplica automáticamente `best_params` al modelo (clasificación o regresión). Elegir método con `--hpo-method=grid|bayes`.
Comprobación rápida:
```bash
# Grid regresión
python scripts/run_hpo.py --task reg --method grid --fast --out-dir outputs/hpo_reg_fast
# Bayes clasificación
python scripts/run_hpo.py --task clf --method bayes --fast --bayes-iter 10 --out-dir outputs/hpo_clf_fast --no-leak-check
# Flujo completo con HPO (grid)
python scripts/run_all.py --with-hpo --hpo-method=grid
# Flujo completo con HPO (bayes)
python scripts/run_all.py --with-hpo --hpo-method=bayes
```
Validación: inspeccionar `reports/hpo_summary.md` para tablas y `outputs/hpo_<task>/best.json` para parámetros óptimos aplicados. Reproducibilidad: ajustar semilla en `config/params.yaml` (`hpo.random_state`).

### 8.3 Validación temporal (TimeSeriesSplit / split fijo)
**Oportunidad**: asegurar robustez temporal (evitar fuga hacia el futuro) cuando existe dimensión año.
Implementación: el split fijo está parametrizado en `config/config.py` (train ≤2018, gap 2019, test 2020–2024) y se aplica en `src/preprocessing/clean.py::temporal_split`. Además, el CV temporal está soportado vía `src/data/splits.py::get_cv` activándolo con `cv.kind: time` en `config/params.yaml`; en HPO se respeta el orden temporal si se pasa `--date-col`.
Comprobación:
- Flujo principal: `python scripts/run_all.py` genera `data/processed/X_train_engineered.csv` y `X_test_engineered.csv` ya separados temporalmente (train ≤2018, test ≥2020).
- HPO con CV temporal: define en `config/params.yaml` `cv.kind: time` y ejecuta, por ejemplo:
```bash
python scripts/run_hpo.py --task reg --method grid --fast --date-col 'AÑO' --out-dir outputs/hpo_reg_time
```
Reconfiguración: ajusta años en `config/config.py` (TRAIN_END_YEAR, TEST_START_YEAR, etc.) y n_splits en `config/params.yaml`.

### 8.4 Próximos pasos
- UI: exponer en ui/app.py un panel “Experimentos (HPO)” con controles para método (grid/bayes), tarea (reg/clf), fast, n_jobs, bayes-iter, y botón Ejecutar que llame a scripts/run_hpo.py y renderice reports/hpo_summary.md y tablas results.csv.
- UI: añadir switch “Validación temporal (TimeSeriesSplit)” que lea cv.kind de config/params.yaml y, en datasets externos, campo para --date-col; mostrar el rango train/test efectivo desde config/config.py.
- Guardado de modelo: tras run_all.py con --with-hpo, persistir modelo final con best_params en models/ (pickle) y mostrar link en UI.
- Trazabilidad: registrar en outputs/metadata/run.json versiones de librerías y método de HPO usado; mostrarlo en UI como metadata de experimento.
- Reproducibilidad: permitir fijar semilla global desde UI y exponer valor actual (config.hpo.random_state).
- Documentación: agregar nota de buenas prácticas (evitar data leakage, usar split temporal) en la sección de ayuda de la UI.

## 9) Implementación EvM5U3

### 9.1 Agrupamiento (Clustering)
Esta implementación (HU-CLUST-01) agrega un flujo de análisis no supervisado para explorar patrones en los datos procesados (X_train_engineered). Se aplican KMeans y DBSCAN sobre un subconjunto escalado (y opcionalmente reducido con PCA) para identificar grupos potenciales de programas/modalidades.

Ejecutar flujo principal (solo EDA + preprocesamiento + entrenamiento rápido, omitiendo evaluación y XAI para acelerar):
```bash
python3 scripts/run_all.py --skip-eval --skip-xai
```

Ejecutar clustering (genera metrics, resumen y gráficos):
```bash
python3 scripts/run_clustering.py --save-plots --pca --max-features 40 --seed 42 --sil-sample 3000 --sample-size 40000
```
Parámetros clave:
- --save-plots: guarda gráficos de Silhouette (KMeans) y heatmap eps/min_samples (DBSCAN) en reports/.
- --pca: aplica PCA si hay >50 features (reduce a <=10 componentes para velocidad).
- --max-features 40: limita columnas iniciales para reducir dimensionalidad y ruido.
- --seed 42: fija reproducibilidad en submuestreos y algoritmos.
- --sil-sample 3000: submuestreo para calcular Silhouette más rápido (si el dataset es grande).
- --sample-size 40000: toma hasta 40k filas para acelerar clustering (si hay más, selecciona aleatoriamente).

Resultados (reports/):
- clustering_results.csv: tabla con algoritmo, parámetros, silhouette, homogeneity, completeness (estas últimas NaN si no hay proxy de etiqueta) y tiempo.
- clustering_summary.md: top 3 configuraciones por Silhouette y total de configuraciones evaluadas.
- clustering_kmeans_silhouette.png y clustering_dbscan_eps_grid.png: visualización comparativa de calidad de agrupamiento.
Interpretación:
- Silhouette mide separación interna de clusters (≈0.2 indica estructura débil/moderada; valores negativos señalan mala asignación).
- DBSCAN eps=0.7 mostró mejor cohesión (Silhouette≈0.22) frente a KMeans clásico, sugiriendo densidades locales aprovechables.
- Homogeneity/completeness se reportan si existe y_train (proxy); de lo contrario se enfocan en métricas intrínsecas.


### 9.2 Detección de Anomalías (Anomaly Detection)
Esta implementación (HU-ANOM-01) incorpora algoritmos no supervisados para identificar registros potencialmente atípicos en el dataset procesado. Permite señalar casos extremos para auditoría, limpieza adicional o generación de reglas.

Ejecutar flujo previo mínimo (genera X_train_engineered):
```bash
python3 scripts/run_all.py --skip-eval --skip-xai
```
Ejecutar detección de anomalías:
```bash
python3 scripts/run_anomaly_detection.py --save-plots --pca --max-features 40 --seed 42 --sample-size 40000 --contamination 0.05
```
Parámetros clave:
- --save-plots: guarda barplot de fracción de anomalías y distribuciones de scores por algoritmo.
- --pca: aplica PCA si >50 columnas para reducir dimensionalidad (<=10 componentes) y acelerar.
- --max-features 40: limita variables iniciales (primeras columnas) para reducir ruido.
- --seed 42: reproducibilidad en submuestreos e inicializaciones.
- --sample-size 40000: submuestrea filas si el dataset es mayor (acelera cómputo en LOF/IsolationForest).
- --contamination 0.05: proporción esperada de anomalías; usada por IsolationForest, LOF, EllipticEnvelope y OneClassSVM.

Resultados (reports/):
- anomaly_results.csv: resumen por algoritmo (anomaly_fraction observada vs contamination_cfg, estadísticas de score, tiempo).
- anomaly_summary.md: top 3 algoritmos cuya fracción de anomalías más se aproxima a la contaminación objetivo.
- anomaly_fraction_by_algo.png: comparación visual de las fracciones detectadas.
- anomaly_scores_<ALGO>.csv / anomaly_scores_dist_<ALGO>.png: scores individuales y su distribución para análisis posterior.
Interpretación:
- anomaly_fraction cercana a contamination_cfg indica calibración adecuada; valores muy altos/bajos sugieren revisar parámetros.
- Scores más extremos (colas) señalan candidatos a inspección manual; comparar entre algoritmos reduce falsos positivos.



### Borrar todos los artefactos y ejecutar el flujo completo
ejecuta comando bash:
```bash
bash scripts/run_full_flow.sh
```

Ejecutar paso a paso
```bash
- bash clean.sh
- python3 scripts/run_all.py --with-hpo --hpo-method=grid
- python3 scripts/run_hpo.py --task clf --method grid --fast --out-dir outputs/hpo_clf_fast
- python3 scripts/run_hpo.py --task reg --method grid --fast --out-dir outputs/hpo_reg_fast_opt
- python3 scripts/run_clustering.py --save-plots --pca --max-features 40 --seed 42 --sil-sample 3000 --sample-size 40000
- python3 scripts/run_anomaly_detection.py --save-plots --pca --max-features 40 --seed 42 --sample-size 40000 --contamination 0.05
```