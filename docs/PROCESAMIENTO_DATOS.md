# Documentación de Preprocesamiento de Datos

Proyecto: Modelado Predictivo para Optimización de Educación Superior en Chile
Ubicación: /home/anaguirv/ia_diplomado/EDA/
Fecha: 2025-11-13T04:37:53.140Z

Objetivo
- Dejar el dataset limpio, consistente y listo para modelado, evitando data leakage y minimizando el uso de memoria/dimensionalidad.

Flujo ejecutado (Paso 3 de la UI)
1) Perfilado inicial
- Identificación de variables objetivo (Y): MODALIDAD (clasificación) y PROMEDIO EDAD PROGRAMA (regresión) si están presentes.
- Separación X (features) e Y, conteo de nulos por celda/columna/fila.

2) Eliminación controlada (no destructiva)
- Columnas: se eliminan solo si su porcentaje de nulos supera un umbral configurable (por defecto 50%).
  Justificación: columnas muy incompletas aportan ruido/imputación inestable y elevan dimensionalidad.
- Filas: se eliminan solo si (a) Y está nula o (b) la fila supera el umbral de nulos por fila (por defecto 80%).
  Justificación: preserva la mayor parte de los datos y evita vaciar el dataset por usar dropna() indiscriminado.

3) División Train/Test previa al ajuste
- train_test_split antes de cualquier “fit”, con estratificación por MODALIDAD si está disponible; tamaño test configurable (por defecto 20%).
  Justificación: evita data leakage al ajustar imputadores/transformadores únicamente con train.

4) Imputación
- Numéricas: SimpleImputer con estrategia configurable (por defecto median), alternativa: mean/most_frequent/constant.
- Categóricas: SimpleImputer con most_frequent o constant (valor por defecto “Desconocido”).
  Justificación: reemplaza nulos sin eliminar observaciones y mantiene consistencia; el ajuste se realiza solo con train.

5) Reducción de cardinalidad en categóricas
- Alta cardinalidad (>100 categorías únicas): frequency encoding (mapea categoría → frecuencia relativa en train). Se aplica el mismo mapeo a test.
  Justificación: reduce drásticamente columnas tras codificación y evita matrices enormes; no usa la variable objetivo (no hay target leakage).
- Categorías raras (<0.5% de frecuencia en train): se agrupan como “__RARE__” en train y test.
  Justificación: controla la explosión de columnas One-Hot y mejora la robustez.

6) Codificación y escalado
- Categóricas restantes: OneHotEncoder(handle_unknown="ignore") con salida sparse (por defecto), forzando salida dispersa a nivel de ColumnTransformer (sparse_threshold=1.0).
  Justificación: memoria eficiente; ‘ignore’ evita fallos por categorías nuevas en test.
- Numéricas: StandardScaler (with_mean=True) aplicado únicamente al bloque numérico (denso).
  Nota: El escalado no densifica el bloque categórico porque permanece sparse.

7) Selección de características
- VarianceThreshold(threshold=1e-6) sobre la matriz transformada (compatible con sparse) para eliminar columnas casi constantes.
  Justificación: remueve señales nulas/redundantes y reduce dimensionalidad sin supervisión adicional.

8) Tipos y memoria
- Conversión del set transformado a float32 (cuando aplica) para reducir ~50% de uso de RAM respecto a float64.
- Estimación de memoria real de matrices sparse (data + indices + indptr) y alerta si supera 8 GB.
  Justificación: control preventivo de recursos; guía al usuario a ajustar umbrales/estrategias en la UI.

Parámetros (por defecto y justificativos)
- Umbral columnas nulas: 50% (equilibrio entre información y ruido).
- Umbral fila nula: 80% (evita eliminar filas por valores dispersos faltantes).
- Tamaño test: 20% (evaluación estable, sin reducir demasiado el train).
- Estrategia numéricas: median (robusta a outliers).
- Estrategia categóricas: most_frequent (consistente); alternativa constant="Desconocido" si hay muchos nulos.
- Rare threshold: 0.5% (reduce columnas sin perder categorías relevantes).
- High-cardinality threshold: 100 categorías (dispara frequency encoding para contener dimensionalidad).
- VarianceThreshold: 1e-6 (umbral conservador para detectar casi-constantes).

Prevención de Data Leakage
- Split antes de ajustes (fit); imputadores, escalado, one-hot y frequency encoding se ajustan exclusivamente con train y se aplican luego a test.
- OneHotEncoder(handle_unknown="ignore") evita aprendizaje implícito de categorías de test.
- Frequency encoding usa solo frecuencias de train (no target-encoding), evitando fuga de información supervisada.

Trade-offs y riesgos
- Frequency encoding sacrifica identidad de categorías, pero mejora memoria y generalización en alta cardinalidad.
- Agrupar raras puede ocultar patrones de “colas”, pero reduce varianza e inestabilidad.
- VarianceThreshold puede eliminar señales débiles; el umbral es configurable si se requiere.

Cómo comprobar en la UI (Paso 3)
- Verifica: Filas originales > 0, Filas descartadas < 30% (si >30% aparecerá advertencia), Train/Test > 0.
- Observa: Features finales razonables y “Memoria estimada matriz train (GB)” por debajo del umbral.
- Si memoria alta: elevar rare_threshold, aumentar high-cardinality encoding (frequency), o subir umbrales de eliminación de columnas muy nulas.

Relación con src/
- Actualmente la lógica está implementada en la UI para evaluación interactiva; la recomendación es mover estas funciones a src/features/preprocess.py y que la UI solo pase parámetros y visualice resultados.
- En pipeline productivo, persistir artefactos (preprocessor, vt, listas de columnas, mapeos de frecuencias) en data/processed/ y outputs/ para reproducibilidad.

Notas finales
- Todas las decisiones buscan: 1) evitar vaciar el dataset, 2) prevenir leakage, 3) contener memoria/dimensionalidad, 4) mantener interpretabilidad básica.
- Ajustar parámetros según el perfil real del dataset y las restricciones de hardware.
