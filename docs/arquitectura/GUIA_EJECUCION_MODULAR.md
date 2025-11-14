# 🏗️ Guía de Uso: Arquitectura Modular Implementada

## ✅ Implementación Completada

Se ha implementado la **arquitectura modular completa** del proyecto con las siguientes capas:

```
┌─────────────────────────────────────────────┐
│   INTERFACES (3 formas de ejecutar)         │
├─────────────────────────────────────────────┤
│ 1. Notebooks (Jupyter)                      │
│ 2. CLI (Command Line Interface)             │
│ 3. UI Web (Streamlit)                       │
└─────────────────────────────────────────────┘
           ↓↓↓ USAN ↓↓↓
┌─────────────────────────────────────────────┐
│   src/pipeline.py (Orquestador Central)     │
├─────────────────────────────────────────────┤
│ MLPipeline: coordina todos los pasos        │
└─────────────────────────────────────────────┘
           ↓↓↓ USA ↓↓↓
┌─────────────────────────────────────────────┐
│   src/ (Módulos Reutilizables)              │
├─────────────────────────────────────────────┤
│ • data/cleaner.py    → Cargar, limpiar     │
│ • visualization/eda.py → Visualizaciones   │
│ • config.py          → Configuración        │
│ • (preprocessing, models, evaluation...)    │
└─────────────────────────────────────────────┘
```

---

## 📂 Archivos Creados

### 🔧 Configuración y Core
```
src/
├── config.py                    (NUEVO) - Configuración centralizada
└── pipeline.py                  (NUEVO) - Orquestador central
```

### 📊 Módulos de Datos
```
src/data/
└── cleaner.py                   (NUEVO) - Carga y limpieza robusto
```

### 📈 Visualizaciones
```
src/visualization/
└── eda.py                       (NUEVO) - Gráficos EDA reutilizables
```

### 🎯 Interfaces
```
main.py                          (MEJORADO) - CLI completo
ui/
└── app.py                       (NUEVO) - Streamlit UI
```

---

## 🚀 Cómo Ejecutar

### Opción 1: CLI (Automatización)

#### Mostrar configuración
```bash
python main.py --mode config
```

#### Ejecutar solo EDA
```bash
python main.py --mode eda
```

#### Entrenar modelos
```bash
python main.py --mode train
```

#### Pipeline completo
```bash
python main.py --mode full
```

#### Con rutas personalizadas
```bash
python main.py --mode eda \
  --input data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv \
  --output outputs/custom/
```

#### Modo verbose (DEBUG)
```bash
python main.py --mode eda --verbose
```

---

### Opción 2: Jupyter Notebooks (Desarrollo)

```python
# En notebook: 01_EDA.ipynb

from src.pipeline import MLPipeline

# Crear pipeline
pipeline = MLPipeline()

# Ejecutar solo EDA
pipeline.load_data()
pipeline.explore_data(output_dir='data/processed')

# O todo junto
pipeline = MLPipeline()
report = pipeline.run_eda_only()
```

---

### Opción 3: UI Web (Demo/Stakeholders)

#### Instalar Streamlit
```bash
pip install streamlit
```

#### Ejecutar app
```bash
streamlit run ui/app.py
```

Luego abrir: **http://localhost:8501**

**Características:**
- 📊 Explorar EDA interactivamente
- 🚀 Ejecutar pipeline con un click
- 📋 Ver dataset completo
- 🎯 Analizar variables objetivo
- 📄 Generar reportes

---

## 📝 Ejemplos de Uso

### Ejemplo 1: Ejecutar EDA desde CLI

```bash
python main.py --mode eda
```

**Salida esperada:**
```
✓ Pipeline inicializado
📥 Cargando datos...
✓ Dataset cargado: 218,566 registros × 42 columnas
Memoria: 45.32 MB
🔍 Explorando datos...
🔍 Generando reporte EDA...
✓ Gráfico guardado: outputs/eda/01_target_classification_MODALIDAD.png
✓ Gráfico guardado: outputs/eda/02_target_regression_PROMEDIO EDAD PROGRAMA.png
✓ Gráfico guardado: outputs/eda/03_missing_values.png
✓ Gráfico guardado: outputs/eda/04_correlation_matrix.png
✓ Reporte EDA completado en: outputs/eda
```

**Archivos generados en `outputs/eda/`:**
- `01_target_classification_MODALIDAD.png` - Distribución de modalidad
- `02_target_regression_PROMEDIO EDAD PROGRAMA.png` - Distribución de edad
- `03_missing_values.png` - Valores faltantes
- `04_correlation_matrix.png` - Matriz de correlaciones

---

### Ejemplo 2: Usar Pipeline desde Python

```python
from src.pipeline import MLPipeline
from src.config import Config
import logging

logging.basicConfig(level=logging.INFO)

# Crear pipeline
config = Config()
pipeline = MLPipeline(config)

# Cargar datos
pipeline.load_data()

# Explorar
report = pipeline.explore_data(output_dir='outputs/custom')

# Ver resultados
print(f"Registros: {pipeline.df.shape[0]:,}")
print(f"Variables objetivo: {report.keys()}")
```

---

### Ejemplo 3: Streamlit UI

1. **Abrir la app:**
   ```bash
   streamlit run ui/app.py
   ```

2. **En la interfaz:**
   - Seleccionar modo en sidebar
   - Click en "Cargar Dataset"
   - Click en "Ejecutar EDA"
   - Ver gráficos y análisis

---

## 🔄 Flujo de Trabajo Recomendado

### Para Desarrollo (Data Scientists)
```
1. Usar Jupyter Notebooks
2. Importar de src/
3. Experimentar con datos
4. Escribir código modular
```

### Para Automatización (ML Ops)
```
1. Usar CLI (main.py)
2. Ejecutar pipeline completo
3. Guardar modelos y reportes
4. Integrar con CI/CD
```

### Para Stakeholders
```
1. Abrir UI (Streamlit)
2. Interactuar sin código
3. Ver resultados visuales
4. Descargar reportes
```

---

## 📊 Estructura de Salida

```
outputs/
├── eda/
│   ├── 01_target_classification_MODALIDAD.png
│   ├── 02_target_regression_PROMEDIO EDAD PROGRAMA.png
│   ├── 03_missing_values.png
│   └── 04_correlation_matrix.png
├── models/                     (próxima fase)
│   ├── classifier.pkl
│   └── regressor.pkl
└── reporte_final.txt          (próxima fase)
```

---

## 🛠️ Configuración

### Personalizar en `src/config.py`

```python
# Cambiar dataset
config.DATASET_PATH = "datos/otro_dataset.csv"

# Cambiar split
config.TRAIN_TEST_SPLIT = 0.7

# Cambiar variables objetivo
config.TARGET_CLASSIFICATION = "MODALIDAD"
config.TARGET_REGRESSION = "PROMEDIO EDAD PROGRAMA"

# Cambiar directorio de salida
config.OUTPUTS_DIR = Path("mis_resultados/")
```

---

## 🧪 Verificación

### Test 1: CLI Funciona
```bash
python main.py --mode config
```

**✓ Debe mostrar configuración en JSON**

### Test 2: EDA Funciona
```bash
python main.py --mode eda
```

**✓ Debe generar gráficos en `outputs/eda/`**

### Test 3: UI Funciona
```bash
streamlit run ui/app.py
```

**✓ Debe abrir navegador en http://localhost:8501**

---

## 📦 Ventajas de Esta Arquitectura

| Aspecto | Beneficio |
|--------|----------|
| **Reutilización** | Código en `src/` se usa en todas partes |
| **Mantenimiento** | Cambios en un lugar, aplica a todos |
| **Testing** | Fácil escribir tests para cada módulo |
| **Escalabilidad** | Agregar nuevas interfaces sin tocar src/ |
| **Producción** | Código limpio y documentado |
| **Demo** | UI para presentar a stakeholders |
| **Automatización** | CLI para pipelines CI/CD |

---

## 🚀 Próximas Fases

### Fase 2: Feature Engineering & Modelos
```
□ Crear src/preprocessing/transformers.py
□ Crear src/models/classification.py
□ Crear src/models/regression.py
□ Entrenar y guardar modelos
```

### Fase 3: Evaluación e Interpretabilidad
```
□ Crear src/evaluation/metrics.py
□ Crear src/interpretation/xai.py
□ Generar reportes con SHAP
□ Comparar modelos
```

### Fase 4: Testing
```
□ Crear tests/test_data.py
□ Crear tests/test_pipeline.py
□ Crear tests/test_models.py
□ Ejecutar pytest
```

---

## 💡 Tips

### Debugging
```bash
# Modo verbose para ver logs detallados
python main.py --mode eda --verbose
```

### Cambiar path del dataset
```bash
python main.py --mode eda --input data/custom/dataset.csv
```

### Cambiar directorio de salida
```bash
python main.py --mode eda --output resultados/custom/
```

### Ejecutar desde notebook
```python
%cd /home/anaguirv/ia_diplomado/EDA
from src.pipeline import MLPipeline
pipeline = MLPipeline()
pipeline.run_eda_only()
```

---

## ✨ Resumen

La **arquitectura modular** está lista y permite:

✅ **3 formas de ejecutar:** Notebooks, CLI, UI  
✅ **Código reutilizable:** `src/` es el core  
✅ **Fácil de mantener:** Un cambio, múltiples usos  
✅ **Producción ready:** CLI para ML Ops  
✅ **User friendly:** UI para stakeholders  

**¡Ahora puedes continuar con el desarrollo en cualquier interface!** 🚀

---

Documento: `GUIA_EJECUCION_MODULAR.md`  
Fecha: 2025-11-12
