# Proyecto: Análisis Predictivo de Titulaciones Universitarias 2007-2024

## 🎯 Objetivo

Aplicar metodología rigurosa de ciencia de datos al análisis y predicción de titulaciones universitarias en Chile, siguiendo las **10 fases** del proyecto de ML operativo.

---

## 📚 Metodología

Este proyecto sigue la estructura del caso **"Predicción operativa en centros de cultivo de salmón"**, adaptada al dominio educativo:

### Fases del Proyecto

| Fase | Nombre | Estado | Notebook |
|------|--------|--------|----------|
| 0 | Entendimiento del problema y datos | ✅ | `notebooks/fase_00/` |
| 1 | Datos y particiones | ✅ | `notebooks/fase_01/` |
| 2 | Ingeniería de características | ✅ | `notebooks/fase_02/` |
| 3 | Métricas de evaluación | ⏳ | `notebooks/fase_03/` |
| 4 | Modelado baseline | ⏳ | `notebooks/fase_04/` |
| 5 | Entrenamiento del modelo | ⏳ | `notebooks/fase_05/` |
| 6 | Evaluación del modelo | ⏳ | `notebooks/fase_06/` |
| 7 | Optimización y fine-tuning | ⏳ | `notebooks/fase_07/` |
| 8 | Interpretabilidad y ética | ⏳ | `notebooks/fase_08/` |
| 9 | Presentación (Model Card) | ⏳ | `notebooks/fase_09/` |
| 10 | Operación y monitoreo | ⏳ | `notebooks/fase_10/` |

---

## 🗂️ Estructura del Proyecto

```
EDA/
├── data/                          # Datos
│   ├── raw/                       # Datos originales
│   ├── processed/                 # Datos procesados
│   ├── train/                     # Conjunto entrenamiento
│   ├── val/                       # Conjunto validación
│   └── test/                      # Conjunto prueba
├── src/                           # Código fuente modular
│   ├── data/                      # Módulos de datos
│   │   ├── loader.py              # Carga con validación
│   │   ├── splitter.py            # Particiones temporales
│   │   └── preprocessor.py        # Preprocesamiento sin leakage
│   ├── features/                  # Feature engineering
│   │   └── engineer.py            # Creación de features
│   ├── models/                    # Modelos
│   ├── evaluation/                # Evaluación
│   ├── interpretation/            # Interpretabilidad
│   └── monitoring/                # Monitoreo
├── notebooks/                     # Notebooks por fase
│   ├── fase_00/                   # Exploración inicial
│   ├── fase_01/                   # Particiones
│   └── ...                        # Una carpeta por fase
├── outputs/                       # Resultados
│   ├── figures/                   # Gráficos
│   ├── models/                    # Modelos guardados
│   ├── reports/                   # Reportes
│   └── tables/                    # Tablas de resultados
├── docs/                          # Documentación
│   ├── PROJECT_STRUCTURE.md       # Estructura detallada
│   ├── DATA_DICTIONARY.md         # Diccionario de datos
│   └── MODEL_CARD.md              # Model Card (Fase 9)
└── scripts/                       # Scripts ejecutables
    └── train.py                   # Entrenamiento
```

---

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Exploración Inicial (Fase 0)

```bash
# Cargar y explorar datos
python -c "from src.data.loader import load_titulados_data; df, _ = load_titulados_data(); print(df.info())"
```

### 3. Crear Particiones (Fase 1)

```python
from src.data.loader import load_titulados_data
from src.data.splitter import split_titulados_data

# Cargar datos
df, _ = load_titulados_data()

# Dividir: 2007-2022 (train), 2023 (val), 2024 (test)
train_df, val_df, test_df = split_titulados_data(df)
```

### 4. Preprocesar (Fase 1)

```python
from src.data.preprocessor import preprocess_titulados_data

# Ajustar en train
train_processed, preprocessor = preprocess_titulados_data(train_df, fit=True)

# Aplicar a val/test (sin leakage)
val_processed, _ = preprocess_titulados_data(val_df, fit=False, preprocessor=preprocessor)
test_processed, _ = preprocess_titulados_data(test_df, fit=False, preprocessor=preprocessor)
```

### 5. Crear Features (Fase 2)

```python
from src.features.engineer import create_titulados_features

# Crear features
train_features, engineer = create_titulados_features(train_processed)
val_features, _ = create_titulados_features(val_processed)
test_features, _ = create_titulados_features(test_processed)
```

---

## 📊 Dataset

**Archivo**: `data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv`

- **Registros**: 218,566 titulaciones
- **Período**: 2007-2024 (18 años)
- **Variables**: 42 columnas
- **Fuente**: Ministerio de Educación de Chile

### Variables Clave

- **Temporal**: AÑO (2007-2024)
- **Geográfica**: REGIÓN, PROVINCIA, COMUNA
- **Institucional**: NOMBRE INSTITUCIÓN, CLASIFICACIÓN
- **Académica**: ÁREA DEL CONOCIMIENTO, NOMBRE CARRERA, NIVEL GLOBAL
- **Target**: TOTAL TITULACIONES (objetivo para regresión/clasificación)
- **Género**: TITULACIONES MUJERES/HOMBRES POR PROGRAMA

Ver `docs/DATA_DICTIONARY.md` para detalles completos.

---

## 🎓 Conceptos Clave (del caso salmoneras)

### Evitar Data Leakage
- ✅ Particiones temporales estrictas (años no se mezclan)
- ✅ Fit solo en train, transform en val/test
- ✅ No usar información del futuro para predecir el pasado

### Estratificación
- Mantener proporciones de clases en train/val/test
- Ejemplo: % por región se mantiene similar

### Imputación
- Calcular estadísticos (media, mediana) solo en train
- Aplicar esos valores a val/test

### Estandarización (z-score)
- `z = (x - media) / desviación`
- Media y desviación calculadas en train únicamente

### Normalización (min-max)
- `x_norm = (x - min) / (max - min)`
- Min y max calculados en train únicamente

---

## �� Análisis Posibles

### Clasificación
1. **Predicción de crecimiento**: ¿Qué instituciones/áreas crecerán?
2. **Clasificación de regiones**: Alto/Medio/Bajo crecimiento
3. **Riesgo de contracción**: Identificar programas en declive

### Regresión
1. **Predicción de titulaciones**: ¿Cuántos titulados habrá?
2. **Demanda futura**: Por área de conocimiento
3. **Impacto de políticas**: Análisis contrafactual

### Series Temporales
1. **Tendencias**: Evolución 2007-2024
2. **Estacionalidad**: Patrones por año
3. **Cambio estructural**: Impacto pandemia (2020-2021)

### Análisis de Equidad
1. **Brecha de género**: Por área y región
2. **Disparidades geográficas**: Acceso por región
3. **Inclusión**: Análisis por tipo de institución

---

## 🛠️ Herramientas

### Desarrollo
- **Python 3.8+**
- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas
- **scikit-learn**: Modelado y preprocesamiento
- **matplotlib/seaborn**: Visualización

### Modelado (próximas fases)
- **XGBoost/LightGBM**: Gradient boosting
- **Prophet**: Series temporales
- **SHAP**: Interpretabilidad

### Monitoreo (Fase 10)
- **MLflow**: Tracking de experimentos
- **Evidently**: Detección de drift

---

## 📝 Convenciones

### Nomenclatura
- Archivos: `snake_case.py`
- Clases: `PascalCase`
- Funciones: `snake_case()`
- Constantes: `UPPER_SNAKE_CASE`

### Git
- Commits descriptivos: `[FASE_X] Descripción clara`
- Branches por fase: `fase_01_particiones`

### Documentación
- Docstrings en todas las funciones públicas
- Type hints cuando sea posible
- Comentarios para lógica compleja

---

## 📖 Referencias

### Documentos del Proyecto
- `docs/PROJECT_STRUCTURE.md`: Estructura completa de las 10 fases
- `docs/DATA_DICTIONARY.md`: Diccionario de datos detallado
- `Evaluación_Proyecto_02.pdf`: Caso de referencia (salmoneras)

### Conceptos Clave
- **Data Leakage**: [Preventing Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- **Time Series CV**: [sklearn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- **CINE-F 2013**: [UNESCO ISCED Fields](http://uis.unesco.org/en/topic/international-standard-classification-education-isced)

---

## ✅ Checklist de Progreso

### Fase 0: Entendimiento ✅
- [x] Cargar datos
- [x] Análisis exploratorio inicial
- [x] Documentar variables
- [x] Identificar problemas de calidad

### Fase 1: Particiones y Preprocesamiento ✅
- [x] Crear módulo de carga
- [x] Implementar particionamiento temporal
- [x] Crear módulo de preprocesamiento
- [x] Validar no hay data leakage

### Fase 2: Feature Engineering ✅
- [x] Features temporales (lags, rolling)
- [x] Features categóricas (STEM, género)
- [x] Features de agregación
- [x] Documentar features creadas

### Fases 3-10: Por Implementar ⏳
- [ ] Fase 3: Definir métricas
- [ ] Fase 4: Modelo baseline
- [ ] Fase 5: Entrenamiento avanzado
- [ ] Fase 6: Evaluación rigurosa
- [ ] Fase 7: Optimización
- [ ] Fase 8: Interpretabilidad
- [ ] Fase 9: Model Card
- [ ] Fase 10: Operación

---

## 🤝 Contribución

Este es un proyecto académico. Para contribuir:

1. Crear branch por fase: `git checkout -b fase_XX_nombre`
2. Implementar según estructura definida
3. Documentar en notebooks
4. Actualizar este README

---

## 📧 Contacto

Proyecto desarrollado para el Diplomado en IA.

**Última actualización**: 2025-10-21
