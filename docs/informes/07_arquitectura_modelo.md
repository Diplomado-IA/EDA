# Arquitectura del modelo (Fase 4)

Objetivo: clasificar MODALIDAD_BIN y apoyar la regresión de PROMEDIO_EDAD_PROGRAMA sobre un dataset tabular con features ingenierizadas.

1) Arquitectura MLP (clasificación)
- Implementación: src/models/model_architecture.py
- Modelo: scikit-learn MLPClassifier dentro de un Pipeline con StandardScaler
- Capas ocultas: 128 -> 64
- Activación: ReLU
- Regularización: L2 (alpha=1e-4) + early_stopping=True (paciencia 10)
- Optimizador: Adam, learning_rate=adaptive, max_iter=200
- Nota: scikit-learn no soporta BatchNorm/Dropout; se sustituye por L2 + early stopping como técnicas de regularización efectivas para tabulares.

2) Features de Fase 4
Generadas en src/preprocessing/clean.py (engineer_features):
- Dummies: PLAN_*, JORNADA_* (alineadas entre train/test)
- Flag POST_2020: 1 si AÑO >= 2020
- Ratios seguros: p.ej. RATIO_TITULACIONES_DURACION, RATIO_EDAD_DURACION (con protección división por 0)
- Temporales por institución: lags (1,2) y media móvil 3 períodos de TOTAL TITULACIONES
- Agregaciones por región y año: sum y mean de TOTAL TITULACIONES
- Índices: HHI_GLOBAL (Herfindahl sobre PROGRAMA), LQ_PROGRAMA (cuociente de localización), IPG_EDAD (Mujer/Hombre)

3) Artefactos y trazabilidad
- Reporte de features: data/processed/feature_engineering_report.txt
- Columnas de OHE: outputs/eda/resumen/one_hot_columns.txt
- Correlación/VIF: data/processed/correlation_matrix.csv, data/processed/vif_scores.csv (+ *_columns_used.txt)
- Selección de features: data/processed/selected_features.txt

4) Criterios de aceptación (Fase 4)
- POST_2020 presente y correcto
- Al menos 1 ratio calculado sin errores
- Lags/rolling generados sobre TOTAL TITULACIONES
- Agregaciones por REGIÓN y AÑO disponibles
- Documentación de arquitectura y regularización en este archivo

