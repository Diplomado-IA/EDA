# Optimización de Consumo de Recursos en Correlación, VIF y Codificación

Resumen
- Por qué sube la CPU: la matriz de correlaciones escala como O(p²·n) y el VIF requiere p regresiones o una inversión matricial; con muchas columnas (p) o filas (n) es costoso. Además, NumPy/BLAS usa múltiples hilos, por lo que puedes ver >100% de CPU (varios núcleos).
- Qué hicimos: aplicamos muestreo de filas, reducción previa de variables y tipos numéricos más ligeros, y un cálculo de VIF más estable. También robustecimos el One‑Hot Encoding para evitar explosiones de memoria.

Cambios implementados en el proyecto (Python/pandas)
1) One‑Hot Encoding seguro en memoria
- Archivo: src/preprocessing/clean.py :: one_hot_encode
- Mejora:
  - Agrupa categorías raras (<0.5%) en "__RARE__" (rare_thresh=0.005).
  - Para columnas de alta cardinalidad (>100 categorías), aplica frequency encoding en vez de OHE.
  - OHE solo en el resto; alinea train/test y tipa dummies a uint8.
  - Excluye objetivos de la codificación.
- Beneficio: evita crear miles de columnas y el OOM killer en WSL.

2) Matriz de correlación más eficiente
- Archivo: src/preprocessing/clean.py :: correlation_matrix
- Mejora:
  - Limita a 200 columnas de mayor varianza (max_cols=200).
  - Muestrea hasta 20.000 filas (sample_rows=20000).
  - Convierte a float32 para reducir memoria y CPU.
- Beneficio: reduce coste O(p²·n) y genera el mismo insight con una muestra representativa.

3) Cálculo de VIF optimizado y estable
- Archivo: src/preprocessing/clean.py :: compute_vif
- Mejora:
  - Excluye objetivos; muestreo de hasta 20.000 filas.
  - Filtra varianza casi cero (threshold=1e-5) y limita a 200 columnas de mayor varianza.
  - Elimina pares con |r|>0.90 para estabilizar inversión.
  - Usa pseudo-inversa de la matriz de correlación (VIF ≈ diag(inv(R))).
  - Downcast a float32.
- Beneficio: menos CPU/memoria y menos problemas de singularidad.

4) Límites de hilos (opcional, a nivel de entorno)
- Para no saturar la máquina, puedes limitar hilos BLAS en la consola antes de ejecutar Streamlit/CLI:
  - Linux/WSL:
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    streamlit run ui/app.py

Cómo usar en la UI
- Fase 3 → pestaña "Correlación & VIF": ahora usa por defecto muestreo/optimizaciones. Se generan:
  - data/processed/correlation_matrix.csv
  - data/processed/vif_scores.csv
- Fase 3 → "Codificar": el botón One‑Hot Encoding aplica rare grouping y frequency encoding automáticamente; escribe outputs/eda/resumen/one_hot_columns.txt.

Buenas prácticas aplicadas (y recomendadas)
- Filtrar antes de calcular: evita IDs, texto libre y columnas con muchísimos NA; considera trabajar por bloques si p es muy grande.
- Usar muestra para diagnóstico: 20k–50k filas suelen bastar para detectar multicolinealidad.
- Tipos ligeros: float32/uint8 donde corresponda.

Interpretación de resultados (umbrales orientativos)
- Correlación (coef. de Pearson |r|):
  - <0.3 baja, 0.3–0.5 moderada, 0.5–0.7 alta moderada, ≥0.7 muy alta.
  - Para predictores: si |r|≥0.7–0.8, considera eliminar/combinar.
- VIF (Variance Inflation Factor):
  - 1: sin colinealidad; ≤5: aceptable; >5: alerta; >10: problemático.
  - Tolerancia = 1/VIF: <0.2 alerta; <0.1 problema serio.

Validación esperada
- Reducción notable de CPU y RAM al calcular correlación y VIF.
- Artefactos generados en las mismas rutas; previsualización en la UI.

Reproducibilidad
- Semilla fija: Config.RANDOM_STATE=42.
- El muestreo afecta solo al diagnóstico (correlación/VIF), no al guardado final de datasets ni al entrenamiento.

Apéndice: Rutas y funciones
- Correlación: src/preprocessing/clean.py::correlation_matrix → data/processed/correlation_matrix.csv
- VIF: src/preprocessing/clean.py::compute_vif → data/processed/vif_scores.csv
- OHE seguro: src/preprocessing/clean.py::one_hot_encode → outputs/eda/resumen/one_hot_columns.txt
