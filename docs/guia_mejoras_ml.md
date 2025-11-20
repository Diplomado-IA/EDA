# Guía para implementar oportunidades de mejora en el modelo (EDA)

Esta guía describe, en formato de historias de usuario adaptadas a ML, cómo implementar tres mejoras clave: (1) control de fuga/tautología del target, (2) tuning sistemático de hiperparámetros y (3) validación temporal. Incluye criterios de aceptación en Gherkin y tareas de programación paso a paso.

---

## HU1 — Evitar fuga/tautología del target

Como científicx de datos
Quiero detectar y mitigar variables que son componentes directos del target o lo determinan casi por completo
Para que el modelo aporte señal nueva, mejore su validez para insight/planificación y evite sobreestimar desempeño

### Contexto del caso
- Ejemplo: predecir PROMEDIO_EDAD_PROGRAMA usando PROMEDIO_EDAD_HOMBRE y PROMEDIO_EDAD_MUJER (95.96% explicado).  
- Alternativas: (a) excluir variables que componen directamente el target, (b) redefinir el target como desviación respecto a una base (p. ej., desviacion_edad = PROMEDIO_EDAD_PROGRAMA − PROMEDIO_EDAD_BASE).

### Criterios de aceptación (Gherkin)
Feature: Control de fuga/tautología en regresión
  Background:
    Given existe un dataset con columnas de entrada y un target definido
  
  Scenario: Detección automática de fuga por componentes del target
    Given el target actual está definido como una agregación de entradas o está altamente colineado
    When calculo R2 de una regresión lineal simple target ~ X_sospechosas
    Then si R2 >= 0.90 se marca leakage_flag = true
    And se registra una advertencia en logs y en un reporte en reports/leakage_report.json

  Scenario: Mitigación por exclusión de variables
    Given leakage_flag = true y config.leakage.strategy = "drop_features"
    When construyo el conjunto de features
    Then se excluyen del entrenamiento las variables marcadas como componentes del target
    And el pipeline se ejecuta sin esas columnas

  Scenario: Mitigación por redefinición del target
    Given leakage_flag = true y config.leakage.strategy = "redefine_target"
    And existe una definición de target_base en la config
    When redefino target := target - target_base
    Then el entrenamiento usa el target redefinido
    And se guarda el mapeo y la fórmula de reconstrucción del target original para inferencia y reporting

  Scenario: Fail-fast si no hay estrategia definida
    Given leakage_flag = true y no hay estrategia seleccionada
    When inicio el entrenamiento
    Then el pipeline falla con un error explicativo y no se entrena ningún modelo

### Tareas de programación (paso a paso)
1) Configuración
- Añadir en config/params.yaml (o equivalente):
  - leakage: { detect: true, r2_threshold: 0.90, strategy: "drop_features" | "redefine_target" | "fail", suspect_features: ["PROMEDIO_EDAD_HOMBRE","PROMEDIO_EDAD_MUJER"], target_base: "PROMEDIO_EDAD_BASE" }

2) Detección
- Implementar en src/features/leakage.py:
  - función detect_leakage(df, target, suspect_features, r2_threshold) -> {flag, r2, selected}
  - usa regresión lineal (sklearn) para R2 y registra métricas.
- Guardar reporte JSON en reports/leakage_report.json con {r2, features_probadas, flag, estrategia}.

3) Mitigación (exclusión de features)
- En el constructor de features (p. ej., src/features/build_features.py), si strategy == drop_features y flag, eliminar columnas sospechosas antes del split/train.

4) Mitigación (redefinición del target)
- En src/data/targets.py, implementar redefine_target(df, target, base_col) -> df[target_red]
- Persistir en outputs/metadata/target_mapping.json la fórmula para reconstrucción y la columna base usada.
- Ajustar inferencia para sumar nuevamente la base si se requiere reportar en la escala original.

5) Salvaguardas
- Añadir aserciones en el pipeline para prohibir que el target esté en X y para verificar fuga básica: abs(corr(X_i, y)) demasiado alto o R2 con sospechosas > umbral.
- Registrar logs claros en outputs/logs/*.log.

6) Documentación
- Actualizar README.md y docs/ con la estrategia seleccionada y riesgos conocidos.

---

## HU2 — Tuning explícito y reproducible de hiperparámetros

Como científicx de datos
Quiero ejecutar un experimento sistemático de hiperparámetros (Grid/Bayes)
Para seleccionar el mejor conjunto con evidencia reproducible, tabla comparativa y artefactos persistidos

### Criterios de aceptación (Gherkin)
Feature: Búsqueda de hiperparámetros reproducible
  Background:
    Given existe un estimador definido y un espacio de hiperparámetros configurado
  
  Scenario: GridSearchCV con CV estratificado/por tiempo según corresponda
    Given config.hpo.method = "grid" y config.cv definido
    When ejecuto la búsqueda con scoring=["neg_mean_absolute_error","neg_root_mean_squared_error","r2"]
    Then se guarda una tabla completa de resultados en outputs/hpo/results.csv
    And se persisten best_params_, best_score_ y la métrica objetivo en outputs/hpo/best.json

  Scenario: Búsqueda bayesiana/Optuna (opcional)
    Given config.hpo.method = "bayes"
    When ejecuto N trials con semilla fija
    Then se genera un study con historial de trials en outputs/hpo/optuna_study.db y un summary en outputs/hpo/optuna_summary.csv

  Scenario: Reproducibilidad y límites de costo
    Given config.hpo.random_state y config.hpo.max_time/min_trials
    When corre la búsqueda
    Then los resultados son reproducibles y el job respeta los límites de tiempo/recursos

### Tareas de programación (paso a paso)
1) Configuración
- En config/params.yaml:
  - hpo: { method: "grid"|"bayes", metric: "neg_root_mean_squared_error", cv: { kind: "kfold"|"time", n_splits: 5, shuffle: false }, grid: { n_estimators: [200,400,800], max_depth: [None,6,10,14], min_samples_split: [2,5,10], min_samples_leaf: [1,2,4] }, bayes: { n_trials: 50, timeout: 3600, random_state: 42 } }

2) Implementación GridSearchCV
- En src/models/train.py:
  - función run_grid_search(estimator, param_grid, X, y, cv, scoring, n_jobs=-1, verbose=0)
  - guardar cv_results_ como CSV y best_* como JSON en outputs/hpo/.

3) Implementación Bayesiana (opcional)
- Crear src/models/hpo_optuna.py con un objetivo que evalúe CV, registre métricas y devuelva pérdida.
- Persistir study (RDB SQLite) y tablas de trials.

4) Reporting
- Generar notebook/report en reports/hpo_report.ipynb o .md con tabla comparativa y selección óptima.
- Graficar importancia de hiperparámetros si se usa Optuna.

5) Integración
- Parametrizar el estimador (RandomForest, XGBoost, etc.).
- Asegurar random_state fijo y que el mismo CV se use en HPO y en la evaluación final.

---

## HU3 — Validación temporal (si aplica dimensión año)

Como científicx de datos
Quiero validar con esquemas temporales (TimeSeriesSplit o cortes por año)
Para asegurar robustez en despliegue y evitar fuga por tiempo

### Criterios de aceptación (Gherkin)
Feature: Validación temporal
  Background:
    Given el dataset contiene una columna temporal (p. ej., fecha o año)
  
  Scenario: Backtesting con TimeSeriesSplit
    Given config.cv.kind = "time" y config.cv.n_splits = K
    When entreno con TimeSeriesSplit(rolling) sin mezclar temporalidad
    Then no hay registros de test en el pasado usados para entrenar
    And se reportan métricas por fold y promedio en outputs/metrics/temporal_cv.csv

  Scenario: Hold-out cronológico
    Given config.split.train_end_year <= 2018 y test_range = 2019-2024
    When realizo el split por fecha
    Then el modelo se entrena solo con datos <= 2018 y se evalúa en 2019-2024
    And se guarda un reporte con MAE, RMSE, R2 por año en outputs/metrics/holdout_temporal.csv

  Scenario: Integración con HPO
    Given config.cv.kind = "time"
    When ejecuto HPO
    Then la misma partición temporal se utiliza para comparar candidatos

### Tareas de programación (paso a paso)
1) Configuración
- En config/params.yaml:
  - cv: { kind: "time", n_splits: 5, gap: 0, max_train_size: null }
  - split: { date_col: "anio"|"fecha", train_end: "2018-12-31" | 2018, test_start: "2019-01-01" | 2019 }

2) Implementación
- En src/data/splits.py:
  - función chronological_split(df, date_col, train_end, test_start) -> (X_train,y_train,X_test,y_test)
  - función get_cv(cv_cfg) que devuelve TimeSeriesSplit con parámetros.

3) Métricas y reporting
- Calcular por fold: MAE, RMSE, R2, MdAPE; guardar por-fold y promedio.
- Generar gráficos de desempeño por tiempo (opcional) en reports/.

4) Salvaguardas
- Asegurar que ninguna transformación basada en estadísticas (scaler, imputador) se ajusta con datos futuros: envolver en Pipeline y fit por fold.

---

## Recomendaciones transversales
- Métricas: usar MAE, RMSE y R2; reportar intervalos por bootstrap si es posible.
- Semillas y reproducibilidad: fijar random_state, versionar datos o hashes, y persistir artefactos (modelos, configs, resultados HPO).
- Trazabilidad: cada corrida debe guardar config usada, commit SHA y timestamp en outputs/metadata/run.json.
- Automatización: exponer tareas como scripts Makefile o bash (train, hpo, report).

## Checklist final (DoD)
- [x] HU1 detección y mitigación documentada (flag=false en datos actuales, R²≈0.19).

- [ ] Existe detección automática de fuga con reporte y estrategia aplicada o fail-fast documentado.
- [ ] Hay resultados de HPO persistidos, con mejor configuración y tabla comparativa.
- [ ] La validación temporal está implementada cuando corresponde, con métricas por corte.
- [ ] Reproducibilidad demostrada (semillas, artefactos, configs, commit SHA).
- [ ] Documentación actualizada en docs/ y README.
