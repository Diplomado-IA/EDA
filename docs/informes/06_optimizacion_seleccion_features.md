# Optimización de la Selección de Features (Selección & Guardado)

Resumen
- Causa del alto consumo: mutual information sobre todas las columnas/filas numéricas escala con p y n y puede ser lento y demandante en CPU/RAM.
- Solución aplicada: muestreo de filas, filtrado por varianza, límite de columnas por mayor varianza, tipos ligeros y ANOVA (f-statistics) por defecto; fallback a mutual information.

Implementación en el proyecto
- Archivo/función: src/preprocessing/clean.py :: select_features
- Cambios clave:
  - sample_rows=20000: muestrea hasta 20k filas para el scoring.
  - var_thresh=1e-6: elimina variables casi constantes.
  - max_cols=500: mantiene como máximo 500 columnas (las de mayor varianza) para el scoring.
  - Downcast a float32 y limpieza de inf/NaN.
  - method="auto": usa f_classif (clasificación) o f_regression (regresión), mucho más rápido; si falla, cae a mutual_info_*.
- Salida: guarda data/processed/selected_features.txt con las variables seleccionadas (top_k=30 por defecto).

Uso en la UI
- Fase 3 → pestaña "Selección & Guardado" → botón "Seleccionar features": emplea la versión optimizada sin cambios en la interacción del usuario.
- Ajustes avanzados: para modificar parámetros (p. ej., top_k, sample_rows, max_cols), edita la llamada a select_features en ui/app.py o en el pipeline según tus necesidades.

Efectos esperados
- Reducción notable del tiempo de cómputo (segundos en lugar de minutos) y menor uso de CPU/RAM.
- Resultados estables para ranking de features en datasets grandes gracias al muestreo estratificado por índice.

Apéndice
- Funciones relacionadas: f_classif/f_regression (sklearn), mutual_info_classif/mutual_info_regression.
- Artefactos: data/processed/selected_features.txt.
