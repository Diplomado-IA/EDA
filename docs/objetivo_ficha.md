# Ficha de Objetivos del Proyecto ML 

## 1. Problema Estratégico
Transformar datos históricos de educación superior (2007–2024) en inteligencia accionable para: (a) anticipar modalidad operativa de programas (optimización de capacidad y costos) y (b) estimar edad promedio de titulación (diseño de apoyos y planificación curricular).

## 2. Variables Objetivo (Y)
- Clasificación: MODALIDAD_BIN (derivada de MODALIDAD: Presencial=1, No presencial/A Distancia=0) [se crea en preprocesamiento].
- Regresión: PROMEDIO EDAD PROGRAMA  (columna original con espacio final, convertido a float).

## 3. Variables Explicativas (X) iniciales (tabular)
- Área CINE (CINE-F_13 Área / Subárea) y Área Carrera Genérica
- Región, Comuna, Sede
- Jornada, Tipo de Plan
- Duración estudio y duración total (semestres)
- Tamaño del programa (TOTAL TITULACIONES, desagregados por género; construcción futura de HHI, IPG, LQ)
- Año (para split temporal y posible drift)
- Institución (nivel 1/2/3 clasificación)
- Rangos de edad (para ingeniería de features agregadas y calidad).

## 4. Métricas de Éxito (Config.METRICS)
- Clasificación: AUC_PR, F1_macro, Brier (calidad probabilística / calibración futura).
- Regresión: MAE (principal), MedAE (robusta), RMSE (penaliza errores grandes).

## 5. Estrategia de Split Temporal
- Train: 2007–2018
- Gap: 2019 (no usado para evitar leakage transición post-pandemia)
- Test: 2020–2024
Representación en config: TRAIN_TEST_SPLIT = "2007-2018 | 2020-2024 (gap 2019)".

## 6. Riesgos (Config.RISKS)
- desbalance_modalidad (clase minoritaria No presencial)
- nulidad_rangos_edad (altos NA en distribuciones etarias)
- drift_post_2020 (cambios estructurales modalidad por pandemia)
- data_leakage (uso indebido de variables posteriores al año objetivo).

## 7. Decisiones Técnicas Iniciales
- Generar MODALIDAD_BIN en preprocesamiento antes de splits.
- Imputación: numérica=median, categórica=most_frequent, rangos de edad vacíos=0.
- Conversión PROMEDIO EDAD PROGRAMA  a float (reemplazo coma -> punto).
- Escalado posterior (z-score) para modelos sensibles si aplica (MLP / ElasticNet).
- Evaluar class_weight y optimización de umbral por costo (Recall clase No presencial) tras calibración.
- Guardar artefactos en outputs/, modelos/ y reports/ según Config.

## 8. Modelos Candidatos (Config)
- Clasificación: LogReg, CatBoost, XGBoost (calibración isotónica opcional).
- Regresión: ElasticNet, CatBoost, LightGBM (intervalos: quantile/conformal en fase posterior).

## 9. Regularización / Generalización
- USE_EARLY_STOPPING = True (CatBoost/XGB/LightGBM)
- USE_WEIGHT_DECAY = True (para MLP futuro)
- USE_DROPOUT = True (en arquitectura MLP futura)

## 10. Indicadores de Calidad EDA (Actual)
- Filas: ~218K
- Columnas: 42
- Nulos totales: ~1.52M (priorizar análisis de columnas con mayor % NA)
- Conversión pendiente de columnas numéricas con coma decimal.

## 11. Criterios de Aceptación Fase 1
- Config refleja dataset grande, métricas extendidas y split temporal.
- Objetivos y riesgos alineados con contexto_proyecto.md.
- Plan claro de creación MODALIDAD_BIN (no error por ausencia inicial).
- Métricas seleccionadas justificadas frente a desbalance y negocio.

## 12. Próximos Pasos (Fase 2)
- Implementar función de ingeniería para creación MODALIDAD_BIN y limpieza edades.
- Generar resumen de nulos por columna y priorizar imputación.
- Construir features derivados (HHI concentración titulaciones, indicadores región, dummies plan/jornada).

## 13. Verificación Rápida
```bash
python -c "from config.config import Config; c=Config(); print(c.TARGET_CLASSIFICATION, c.TARGET_REGRESSION, c.METRICS, c.TRAIN_TEST_SPLIT, c.RISKS)"
```

## 14. Estado
Documento alineado con contexto estratégico y configuración vigente al 2025-11-14T02:58:45.259Z.

