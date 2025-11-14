# Documentación del Proceso de Preprocesamiento de Datos (Proyecto Educación Superior)

Última actualización: 2025-11-14

1. Objetivo

El objetivo del preprocesamiento es preparar los datos para modelado predictivo de forma robusta y eficiente, minimizando pérdida de información y evitando problemas como multicolinealidad, alta cardinalidad y uso excesivo de memoria. El pipeline evita data leakage y busca reproducibilidad total.

2. Flujo General del Proceso

Etapa | Descripción | Resultado
- Limpieza | Imputación de nulos y normalización de objetivos (MODALIDAD_BIN, PROMEDIO EDAD PROGRAMA ) | Dataset sin nulos críticos
- Split temporal | Train ≤2018, gap 2019, Test ≥2020 (configurable en config/config.py) | Conjuntos independientes
- Normalización | StandardScaler para variables numéricas (fit solo en train) | Variables centradas y escaladas
- Codificación | Control de cardinalidad, OHE o frequency encoding según criterio | Representación numérica eficiente
- Reducción | VarianceThreshold y conversión a float32/sparse | Menor dimensionalidad y memoria

3. Detalle de Cada Etapa (adaptado al proyecto)

3.1 Manejo de Valores Faltantes
- Implementación: src/preprocessing/clean.py::impute_values
- Config: Config.IMPUTE_NUM = "median", Config.IMPUTE_CAT = "most_frequent"
- Efecto: conserva tamaño muestral imputando medianas/moda. No se eliminan filas masivamente.
- Artefactos: se reflejan en outputs/eda/resumen/scaler_info.txt tras etapas posteriores.

3.2 Preparación de Objetivos
- MODALIDAD_BIN: si falta y existe MODALIDAD, se crea binaria (presencial=1, otras=0).
- PROMEDIO EDAD PROGRAMA : coaccionado a numérico (to_numeric errors='coerce').

3.3 División Temporal Train/Test
- Implementación: src/preprocessing/clean.py::temporal_split
- Lógica: extrae año numérico desde columna AÑO (p.ej. "TIT_2024").
- Config: TRAIN_START_YEAR=2007, TRAIN_END_YEAR=2018, GAP=2019, TEST_START_YEAR=2020-2024.
- Validación: logs de dtype y ejemplo, máscaras with between; error claro si no hay años válidos.

3.4 Estandarización de Variables Numéricas
- Implementación: src/preprocessing/clean.py::scale_numeric
- Detalle: StandardScaler fit en train, transform en test; exclusión de columnas objetivo.
- Artefacto: outputs/eda/resumen/scaler_info.txt (medias y varianzas)

3.5 Codificación de Categóricas con Control de Memoria
- Implementación actual: one_hot_encode usa pandas.get_dummies(drop_first=True) y alinea columnas.
- Riesgo: para alta cardinalidad puede agotar memoria (WSL). Se recomienda:
  1) Agrupar categorías raras (<0.5%) en "__RARE__" antes de OHE.
  2) Usar OneHotEncoder(handle_unknown='ignore', sparse_output=True) o frequency encoding para cols con >100 categorías.
  3) Excluir IDs/texto libre de la codificación.
- Artefacto: outputs/eda/resumen/one_hot_columns.txt

3.6 Ingeniería de Características
- Implementación: src/preprocessing/clean.py::engineer_features
- Acciones: dummies específicas (PLAN, JORNADA), índices HHI_GLOBAL, LQ_PROGRAMA, IPG_EDAD si hay columnas requeridas.
- Artefacto: data/processed/feature_engineering_report.txt

3.7 Reducción de Dimensionalidad y Calidad
- Implementación: src/preprocessing/clean.py::{correlation_matrix, compute_vif, select_features}
- Acciones: matriz de correlación, VIF para multicolinealidad, selección por MI (top_k=30 por defecto).
- Recomendado: convertir a float32 y aplicar VarianceThreshold(1e-6) si se expande la matriz.
- Artefactos: data/processed/correlation_matrix.csv, data/processed/vif_scores.csv, data/processed/selected_features.txt

3.8 Guardado de Datasets Finales
- Implementación: src/preprocessing/clean.py::save_datasets
- Archivos: data/processed/X_train_engineered.csv, X_test_engineered.csv (sin Y, filtradas por features seleccionadas).

4. Control de Calidad y Reproducibilidad
- Sin leakage: todos los ajustes (scaler, codificadores) se realizan con train.
- Checks: sin NaN tras transformaciones; consistencia de columnas entre train/test.
- Recursos: objetivo <8GB RAM; si hay OOM, aplicar recomendaciones de 3.5 y 3.7.
- Seeds: Config.RANDOM_STATE=42.

5. Cómo Ejecutar
- UI: streamlit run ui/app.py → Fase 3 pestañas Limpieza, Split, Codificar, etc.
- CLI: ./venv/bin/python scripts/execute_pipeline.py --phase 3 (si está disponible).

6. Parámetros Clave (config/config.py)
- DATASET_PATH, SEPARATOR=';', ENCODING='latin1'
- TARGET_CLASSIFICATION='MODALIDAD_BIN', TARGET_REGRESSION='PROMEDIO EDAD PROGRAMA '
- TRAIN/TEST años: 2007-2018 y 2020-2024 (gap 2019)
- IMPUTE_NUM='median', IMPUTE_CAT='most_frequent'

7. Buenas Prácticas para este dataset
- Antes de OHE: revisar cardinalidad de columnas object y agrupar raras.
- Evitar codificar columnas con texto libre o IDs.
- Convertir numéricas a float32 antes de guardar; preferir matrices sparse cuando sea posible.
- Verificar proporción de clases de MODALIDAD_BIN en train/test para drift post-2020.

8. Conclusión
El pipeline de preprocesamiento de este proyecto limpia, divide temporalmente, escala y codifica los datos con foco en reproducibilidad y uso eficiente de memoria. Siguiendo estas instrucciones se evita data leakage y caídas por OOM en WSL, preparando artefactos consistentes para el entrenamiento de modelos.
