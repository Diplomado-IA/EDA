# Paso a paso para construir el proyecto  
*(alineado a las 8 fases y criterios)*

---

## Fase 1: Comprensión

1. Crear ficha de 1 página (`docs/objetivo_ficha.md`) con:
   - Problema.
   - **Y clasificación** = `MODALIDAD_BIN`.
   - **Y regresión** = `PROMEDIO_EDAD_PROGRAMA`.
   - Listado de variables **X**: `CINE`, región, jornada, duración, tamaño programa, año, institución.

2. Definir métricas:
   - **Clasificación** → AUC-PR, F1-macro.
   - **Regresión** → MAE, RMSE.

3. Registrar riesgos:
   - Desbalance de clases.
   - Alta nulidad en edad.
   - *Drift* temporal post-2020.

---

## Fase 2: EDA

4. Implementar `cargar_csv()` en `src/eda.py` con detección de *encoding*.  
5. Generar funciones:
   - `resumen_columnas()`
   - `descriptivos_numericos()`
   - `plot_distrib_objetivo()`

6. Ejecutar `eda_minimo()` y guardar resultados en:
   - `outputs/eda/resumen/*.csv`
   - `outputs/eda/figures/*.png`

7. Calcular *imbalance ratio* y decidir uso de `class_weight`.

---

## Fase 3: Preprocesamiento

8. Implementar en `src/preprocessing/clean.py`:
   - `handle_missing_data()`
   - `convert_to_numeric()`
   - `encode_categoricals()`
   - `scale_numeric()`

9. Definir partición:
   - **Temporal**: `train ≤ 2019`, `test ≥ 2020`.
   - **Estratificada** para `MODALIDAD_BIN` dentro de *train* (generar `X_train`, `X_val`, `X_test` y `y_*`).

10. Persistir datasets en:
    - `data/processed/*.parquet` (con ingeniería previa).
    - Pickles para scripts actuales.

---

## Fase 4: Feature Engineering / Arquitectura

11. Completar `engineer_features()`:
    - Ratios.
    - `POST_2020`.
    - *Dummies*.

12. Añadir:
    - *Features* temporales (lags por institución/año, si aplica).
    - Agregaciones (`sum` / `mean` por región).

13. Diseñar MLP:
    - `build_mlp_classifier()` con arquitectura:
      - Capas: `128 → BatchNorm → Dropout(0.3) → 64 → salida`.

14. Documentar arquitectura y regularización en `docs/arquitectura_modelo.md`:
    - Dropout.
    - L2.
    - EarlyStopping.

---

## Fase 5: Entrenamiento y Evaluación

15. Crear script `training_tabular.py` que:
    - Carga `X_train` / `X_val`.
    - Compila y entrena el modelo con `EarlyStopping`.
    - Guarda `history` + modelo en:
      - `models/trained/mlp_modalidad.pkl` / `.h5`.

16. Entrenar modelos clásicos *baseline*:
    - Logistic Regression (LogReg).
    - Random Forest (RF).
    - Guardar métricas en:
      - `outputs/metrics/classification_metrics.json`.

17. Evaluar en *test*:
    - Matriz de confusión.
    - Curva PR.
    - Umbral óptimo:
      - Máx. F1 **o**
      - Objetivo `Recall ≥ 0.90` para clase minoritaria.
    - Guardar *plots*.

18. Para **regresión**:
    - Entrenar *baselines*: Linear, Ridge, RF, Gradient Boosting (GB) y MLP regresor.
    - Guardar métricas (MAE, RMSE, R²) y comparación con *baseline*.

---

## Fase 6: Interpretabilidad

19. Calcular:
    - **Permutation importance** (RF / GB).
    - **SHAP**:
      - `TreeExplainer` para modelos de árboles.
      - `KernelExplainer` para MLP.

20. Guardar:
    - `feature_importance_classification.csv`
    - `feature_importance_regression.csv`
    - `shap_summary.png`
    - `shap_values_sample.json`

21. Extraer *insights*:
    - Top variables.
    - Confirmación de hipótesis (`POST_2020`, Área CINE).
    - Documentar en `reports/xai_summary.json`.

---

## Fase 7: Documentación y Presentación

22. Generar informe técnico `reports/final_report.md` (luego a PDF) con:
    - Metodología.
    - Datos.
    - Métricas.
    - Interpretabilidad.
    - Riesgos / limitaciones.

23. Crear **Model Card** `reports/model_card.md` con:
    - *Intended use*.
    - *Out-of-scope*.
    - Fairness.
    - Mantenimiento.

24. Preparar visualizaciones:
    - Pérdida train/val.
    - Métricas por subgrupo (ej. región).
    - Matriz de confusión.
    - Curva PR.

---

## Fase 8: Implementación y Recomendaciones

25. Crear script `scoring.py` para inferencia *batch*:
    - Lee nuevos CSV.
    - Genera predicciones con umbral definido.

26. Definir monitoreo:
    - `metrics_log.json`.
    - Script `monitor_drift.py`:
      - Test KS sobre distribución.
      - Alerta si `p < 0.01`.

27. Documentar plan de *retraining*:
    - Cada trimestre **o**
    - Cuando se detecte *drift*.

28. Generar lista de recomendaciones accionables en `docs/recomendaciones.md`:
    - Optimizar oferta según modalidad predicha.
    - Ajustar apoyos según edad proyectada.

---

## Integración del Pipeline

29. Actualizar `src/pipeline.py` para encadenar:
    - `load_data` → `explore_data` → `preprocess_data` → `feature_engineering` → `train` → `evaluate` → `interpret`.

30. Ajustar `scripts/run_pipeline.py` para:
    - Producir todos los artefactos.
    - Retornar *exit code* 0 en caso de éxito.

31. Añadir `scripts/verify_pipeline.py` con *checks* ampliados:
    - Existencia de métricas.
    - Existencia de modelos.
    - Existencia de artefactos de interpretabilidad.

---

## Verificación Final

32. Ejecutar:

    ```bash
    bash clean.sh
    python scripts/run_pipeline.py
    python scripts/step4_interpretability.py
    ```

33. Verificar:

    ```bash
    ls outputs/eda/figures      # ≥ 5 figuras
    cat outputs/metrics/*.json  # métricas generadas
    ls models/trained           # ≥ 2 modelos
    ls reports/*importance*.csv # archivos de importancia de variables
    ```

34. Revisar **Model Card** y reporte PDF; confirmar alineación con objetivos de negocio.

---

## Criterios de cierre e iteración

35. Criterios de cierre:
    - Todos los criterios de aceptación por fase cumplidos.
    - Sin *tracebacks*.
    - Métricas ≥ umbrales establecidos.
    - Documentación completa.
    - Plan de monitoreo definido.

36. Iteración continua:
    - Si las métricas < umbral **o** se detecta *drift*, volver a **Fase 2 (EDA)** y ajustar imputación / ingeniería antes de reentrenar.

---

> Este plan debe implementarse de forma incremental; cada fase produce artefactos verificables que sirven como puntos de control.
