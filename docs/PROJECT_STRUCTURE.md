# Estructura del Proyecto - Análisis Titulados 2007-2024

## Siguiendo metodología del caso Salmoneras

Este proyecto sigue las **10 fases** del caso de predicción operativa en centros de cultivo de salmón, adaptadas al análisis de titulaciones universitarias en Chile.

---

## Fases del Proyecto

### 0. Entendimiento del problema y los datos
**Objetivo**: Comprender el dominio y las variables disponibles

**Actividades**:
- Definir preguntas de negocio
- Exploración inicial de variables (EDA)
- Identificar variables temporales, categóricas y numéricas
- Detectar valores faltantes y outliers

**Entregables**:
- Notebook de exploración inicial
- Diccionario de datos
- Documento de preguntas de negocio

---

### 1. Datos y particiones
**Objetivo**: Preparar datos evitando fugas (data leakage)

**Conceptos clave**:
- **Train/Val/Test split**: División temporal estricta
- **Estratificación**: Mantener proporciones por región/institución
- **Data leakage**: No usar información del futuro
- **Imputación**: Rellenar valores faltantes
- **Estandarización**: Z-score para variables numéricas
- **Normalización**: Min-Max cuando sea necesario
- **Outliers**: Detección y tratamiento

**Actividades**:
- Dividir datos por año (ej: 2007-2022 train, 2023 val, 2024 test)
- Estratificar por región o tipo de institución
- Imputar valores faltantes de forma informada
- Estandarizar variables numéricas

**Entregables**:
- Scripts de preprocesamiento
- Datasets train/val/test guardados
- Reporte de estadísticas por partición

---

### 2. Ingeniería de características (Features)
**Objetivo**: Crear variables derivadas que mejoren el análisis

**Para datos temporales**:
- **Ventana (L)**: Observaciones hacia atrás
- **Horizonte (H)**: Predicción hacia adelante
- **Rezagos (lags)**: Valores pasados
- **Rolling features**: Promedio móvil, tendencias
- **Variación porcentual**: Cambios año a año

**Para datos de titulados**:
- Agregaciones por año/región/institución
- Ratios (mujeres/hombres, titulados/matriculados)
- Tendencias temporales
- Categorías derivadas (áreas STEM vs no-STEM)

**Actividades**:
- Crear features temporales por año
- Calcular tasas de crecimiento
- Agregar por diferentes dimensiones
- One-hot encoding para categóricas

**Entregables**:
- Módulo de feature engineering
- Documentación de features creadas
- Análisis de correlaciones

---

### 3. Métricas de evaluación (Clasificación)
**Objetivo**: Definir cómo medir el éxito

**Métricas clave**:
- **Matriz de confusión**: TP, FP, FN, TN
- **Accuracy**: (TP + TN) / total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo curva ROC
- **AUC-PR**: Área bajo curva Precision-Recall

**Para regresión**:
- **MAE**: Error absoluto medio
- **RMSE**: Raíz del error cuadrático medio
- **R²**: Coeficiente de determinación

**Aplicación**:
- Clasificar regiones con alto/bajo crecimiento
- Predecir deserción estudiantil
- Clasificar áreas en expansión/contracción

**Entregables**:
- Módulo de métricas
- Baseline de comparación
- Análisis de desbalance de clases

---

### 4. Modelado base (Baseline)
**Objetivo**: Establecer modelo simple de referencia

**Modelos baseline**:
- Regresión Logística
- Random Forest simple
- Promedio histórico (para series temporales)
- Persistencia (valor del año anterior)

**Actividades**:
- Entrenar modelo simple
- Evaluar en val/test
- Documentar limitaciones
- Establecer benchmark

**Entregables**:
- Modelo baseline entrenado
- Métricas de referencia
- Análisis de errores iniciales

---

### 5. Entrenamiento del modelo
**Objetivo**: Ajustar modelos más sofisticados

**Técnicas**:
- **Partición estricta**: Por tiempo y entidad
- **Desbalance**: class_weight o sampling
- **Estabilidad**: gradient clipping
- **Registro**: Curvas de pérdida y métricas
- **Early stopping**: Detener cuando val no mejora

**Modelos avanzados**:
- Gradient Boosting (XGBoost, LightGBM)
- Modelos de serie temporal (Prophet, ARIMA)
- Redes neuronales (si aplica)

**Entregables**:
- Modelos entrenados
- Gráficos de curvas train/val
- Checkpoints guardados

---

### 6. Evaluación del modelo
**Objetivo**: Medir desempeño realista

**Actividades**:
- Evaluar en test set (datos nunca vistos)
- Métricas por subgrupos (región, tipo institución)
- Análisis de errores
- Casos difíciles identificados
- Calibración de probabilidades

**Análisis especiales**:
- Desempeño por año
- Desempeño por región
- Desempeño por área de conocimiento
- Identificar patrones en errores

**Entregables**:
- Informe de evaluación completo
- Tablas y gráficas de métricas
- Lista de casos difíciles comentados

---

### 7. Optimización y fine-tuning
**Objetivo**: Mejorar sin romper costos

**Técnicas**:
- **HPO**: Optimización de hiperparámetros
- **Regularización**: Dropout, weight decay
- **Comparativas**: Diferentes algoritmos
- **Ablation**: Medir aporte de cada feature
- **Transfer learning**: Si aplica

**Actividades**:
- Grid/Random search de hiperparámetros
- Probar diferentes features
- Comparar modelos
- Seleccionar mejor configuración

**Entregables**:
- Tabla comparativa de variantes
- Mejor modelo seleccionado
- Análisis costo-beneficio

---

### 8. Interpretabilidad y ética
**Objetivo**: Explicar decisiones del modelo

**Técnicas**:
- **Feature importance**: Variables más influyentes
- **SHAP values**: Explicaciones locales
- **Partial dependence plots**: Efecto de cada variable
- **Casos ejemplo**: Explicar predicciones específicas

**Consideraciones éticas**:
- Sesgos por género/región
- Equidad en subgrupos
- Uso responsable de predicciones
- Limitaciones del modelo

**Actividades**:
- Calcular importancia de variables
- Analizar sesgos por subgrupos
- Documentar limitaciones
- Definir uso apropiado/inapropiado

**Entregables**:
- Sección de interpretabilidad
- Análisis de sesgos
- Documento de consideraciones éticas

---

### 9. Presentación (Model Card)
**Objetivo**: Comunicar resultados y límites

**Contenido del Model Card**:
- **Propósito**: Para qué sirve el modelo
- **Datos**: Qué datos usa y cómo se obtuvieron
- **Entrenamiento**: Algoritmo y proceso
- **Métricas**: Desempeño global y por subgrupos
- **Límites**: Qué NO puede hacer
- **Usos**: Apropiados e inapropiados
- **Próximos pasos**: Roadmap de mejoras

**Presentación**:
- 6-10 slides
- Historia clara: problema → datos → modelo → resultados
- Errores típicos y cómo mitigarlos
- Recomendaciones operativas

**Entregables**:
- Model Card completo
- Presentación en PDF/PPTX
- Video de presentación (opcional)

---

### 10. Operación y monitoreo
**Objetivo**: Usar el modelo en producción

**Actividades**:
- Pipeline de scoring automatizado
- Monitoreo de drift (cambios en distribución)
- Alertas cuando métricas caen
- Proceso de reentrenamiento
- Dashboard de monitoreo

**Para el caso de titulados**:
- Predicciones anuales por institución
- Alertas de caída en titulaciones
- Monitoreo de tendencias
- Actualización con datos nuevos

**Entregables**:
- Plan de monitoreo
- Scripts de scoring
- Dashboard simple
- Procedimiento de actualización

---

## Estructura de Carpetas

```
EDA/
├── data/                          # Datos crudos y procesados
│   ├── raw/                       # Datos originales
│   ├── processed/                 # Datos procesados
│   ├── train/                     # Train set
│   ├── val/                       # Validation set
│   └── test/                      # Test set
├── notebooks/                     # Notebooks por fase
│   ├── 00_exploracion.ipynb       # Fase 0
│   ├── 01_particiones.ipynb       # Fase 1
│   ├── 02_features.ipynb          # Fase 2
│   ├── 03_metricas.ipynb          # Fase 3
│   ├── 04_baseline.ipynb          # Fase 4
│   ├── 05_entrenamiento.ipynb     # Fase 5
│   ├── 06_evaluacion.ipynb        # Fase 6
│   ├── 07_optimizacion.ipynb      # Fase 7
│   ├── 08_interpretabilidad.ipynb # Fase 8
│   └── 09_model_card.ipynb        # Fase 9
├── src/                           # Código modular
│   ├── data/                      # Módulos de datos
│   │   ├── loader.py              # Carga de datos
│   │   ├── preprocessor.py        # Preprocesamiento
│   │   └── splitter.py            # Particiones
│   ├── features/                  # Feature engineering
│   │   ├── temporal.py            # Features temporales
│   │   ├── aggregations.py        # Agregaciones
│   │   └── encoders.py            # Encodings
│   ├── models/                    # Modelos
│   │   ├── baseline.py            # Modelos baseline
│   │   ├── advanced.py            # Modelos avanzados
│   │   └── utils.py               # Utilidades
│   ├── evaluation/                # Evaluación
│   │   ├── metrics.py             # Métricas
│   │   ├── plots.py               # Visualizaciones
│   │   └── reports.py             # Reportes
│   ├── interpretation/            # Interpretabilidad
│   │   ├── feature_importance.py  # Importancia
│   │   └── explainers.py          # SHAP, etc.
│   └── monitoring/                # Monitoreo
│       ├── drift.py               # Detección de drift
│       └── alerts.py              # Sistema de alertas
├── outputs/                       # Resultados
│   ├── figures/                   # Gráficos
│   ├── models/                    # Modelos guardados
│   ├── reports/                   # Reportes
│   └── tables/                    # Tablas de resultados
├── docs/                          # Documentación
│   ├── PROJECT_STRUCTURE.md       # Este archivo
│   ├── DATA_DICTIONARY.md         # Diccionario de datos
│   ├── MODEL_CARD.md              # Model Card
│   └── ETHICS.md                  # Consideraciones éticas
└── scripts/                       # Scripts ejecutables
    ├── train.py                   # Entrenamiento
    ├── evaluate.py                # Evaluación
    └── predict.py                 # Predicción
```

---

## Convenciones de Código

1. **Nomenclatura**:
   - Archivos: snake_case
   - Clases: PascalCase
   - Funciones: snake_case
   - Constantes: UPPER_SNAKE_CASE

2. **Documentación**:
   - Docstrings en todas las funciones
   - Type hints cuando sea posible
   - Comentarios para lógica compleja

3. **Testing**:
   - Tests unitarios en `tests/`
   - Cobertura mínima 70%

4. **Git**:
   - Commits descriptivos
   - Branches por fase
   - Pull requests para revisión

---

## Próximos Pasos

1. ✅ Crear estructura de carpetas
2. ✅ Adaptar notebooks existentes
3. ⏳ Implementar Fase 1: Particiones
4. ⏳ Implementar Fase 2: Features
5. ⏳ Continuar con fases siguientes

---

**Última actualización**: 2025-10-21
