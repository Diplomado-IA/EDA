# 🏗️ ARQUITECTURA DEL PROYECTO

## Índice de Documentación

### Archivos de Referencia:
- `ARQUITECTURA_MODULAR.md` - Diseño modular completo
- `GUIA_EJECUCION_MODULAR.md` - Guía de ejecución paso a paso

### Componentes Principales:

#### **1. Estructura de Directorios**
```
/
├── data/
│   ├── raw/                    # Datos originales
│   └── processed/              # Datos preprocesados
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuración
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── features/
│   │   └── engineer.py
│   ├── models/
│   │   └── trainer.py
│   └── pipeline.py             # Pipeline integrado
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model_Training.ipynb
├── ui/
│   └── app.py                  # Streamlit UI
├── tests/
│   └── test_*.py
├── requirements.txt
└── README.md
```

#### **2. Módulos Principales**

**src/config.py**
```python
# Rutas base
DATA_RAW = 'data/raw/'
DATA_PROCESSED = 'data/processed/'
MODELS_DIR = 'models/'

# Parámetros
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

**src/pipeline.py**
```python
class MLPipeline:
    def load_data()
    def preprocess_data()
    def engineer_features()
    def train_models()
    def evaluate_models()
```

#### **3. Flujo de Ejecución**

```
┌─────────────────────────────────────────┐
│   1. CARGAR DATOS (Jupyter Notebook)    │
│   → notebooks/01_EDA.ipynb              │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   2. PREPROCESAMIENTO (Integrado)       │
│   → src/preprocessing/preprocessor.py   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   3. FEATURE ENGINEERING (Integrado)    │
│   → src/features/engineer.py            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   4. ENTRENAR MODELOS (Integrado)       │
│   → src/models/trainer.py               │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   5. VISUALIZAR RESULTADOS (UI)         │
│   → ui/app.py (Streamlit)               │
└─────────────────────────────────────────┘
```

#### **4. Integración en Pipeline**

```python
from src.pipeline import MLPipeline

# Ejecución automática
pipeline = MLPipeline()
pipeline.load_data()
pipeline.preprocess_data()
pipeline.engineer_features()
pipeline.train_models()
pipeline.evaluate_models()

# Acceso a resultados
pipeline.X_train          # Datos preprocesados
pipeline.preprocessor     # Objeto preprocesador
pipeline.feature_engineer # Objeto feature engineer
pipeline.trainer          # Objeto entrenador
pipeline.models           # Modelos entrenados
pipeline.metrics          # Métricas de evaluación
```

#### **5. Configuración Global**

Ver: `src/config.py`
```python
# Parámetros de preprocesamiento
IMPUTATION_METHOD = 'mean'
OUTLIER_METHOD = 'iqr'
SCALER = 'standard'

# Parámetros de feature engineering
CORRELATION_THRESHOLD = 0.8
VIF_THRESHOLD = 10
N_FEATURES_SELECT = 15
VARIANCE_THRESHOLD = 0.01

# Parámetros de modelos
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
```

---

## 📊 Flujo de Datos

```
CSV (173,522 × 40)
    ↓
EDA (análisis)
    ↓
Preprocesamiento (limpieza)
    ↓
Dataset (173,522 × 40)
    ↓
Feature Engineering (optimización)
    ↓
Dataset Optimizado (173,522 × 15)
    ↓
Train/Test Split
    ↓
X_train (138,818 × 15) | X_test (34,704 × 15)
    ↓
Entrenamiento Modelos
    ↓
Modelos Entrenados + Métricas
    ↓
Visualización (Streamlit UI)
```

---

## 🔄 Componentes Reutilizables

Cada módulo es independiente pero integrado:

### Preprocesador
```python
from src.preprocessing.preprocessor import Preprocessor

prep = Preprocessor()
X_clean = prep.fit_transform(X)
```

### Feature Engineer
```python
from src.features.engineer import FeatureEngineer

engineer = FeatureEngineer()
X_engineered = engineer.fit_transform(X_clean)
```

### Entrenador
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train)
metrics = trainer.evaluate(X_test, y_test)
```

---

## ✅ Validaciones

Cada módulo incluye:
- ✅ Validación de entrada
- ✅ Validación de salida
- ✅ Logging
- ✅ Manejo de errores
- ✅ Tests unitarios

---

## 🚀 Ejecución

Ver `docs/fase0_inicio/QUICK_START.md` para instrucciones completas.

```bash
# Instalación
pip install -r requirements.txt

# Ejecución completa
python -m src.pipeline

# O ejecución paso a paso
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 📚 Documentación Relacionada

- **Fase 1**: `docs/fase1_eda/`
- **Fase 2.1**: `docs/fase2_preprocesamiento/`
- **Fase 2.2**: `docs/fase2_feature_engineering/`
- **Fase 3**: `docs/fase3_modelos/`
- **Integración**: `docs/integracion/`
