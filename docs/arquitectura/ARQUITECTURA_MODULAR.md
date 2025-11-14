# 🏗️ Arquitectura Modular: Notebooks + CLI/UI + Tests

## ✅ Concepto Correcto

No eliminar `src/` sino **organizarlo para múltiples interfaces**:

```
Notebooks (EDA, análisis, desarrollo)
    ↓
src/ (Código reutilizable)
    ↓
main.py (CLI/interfaz)
    ↓
UI/Demo (Streamlit, Flask, etc)
```

---

## 📋 Estructura Recomendada

```
src/
├── __init__.py
├── config.py                    ← Configuración centralizada
├── data/
│   ├── __init__.py
│   ├── loader.py               ← Cargar CSV
│   ├── cleaner.py              ← Limpiar datos
│   └── splitter.py             ← Split train/test
├── preprocessing/
│   ├── __init__.py
│   ├── transformers.py         ← Transformaciones
│   └── validation.py           ← Validaciones
├── features/
│   ├── __init__.py
│   └── engineer.py             ← Feature engineering
├── models/
│   ├── __init__.py
│   ├── classification.py       ← Modelos clasificación
│   └── regression.py           ← Modelos regresión
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              ← Cálculo de métricas
│   └── reports.py              ← Generación reportes
├── interpretation/
│   ├── __init__.py
│   └── xai.py                  ← SHAP, feature importance
├── visualization/
│   ├── __init__.py
│   ├── eda.py                  ← Gráficos EDA
│   └── results.py              ← Gráficos resultados
├── utils/
│   ├── __init__.py
│   ├── logger.py               ← Logging
│   └── helpers.py              ← Funciones auxiliares
└── pipeline.py                 ← Orquestador (ML pipeline)

scripts/
├── __init__.py
├── train.py                    ← Script de entrenamiento
├── predict.py                  ← Script de predicción
└── evaluate.py                 ← Script de evaluación

main.py                          ← CLI principal (MANTENER)

ui/
├── app.py                      ← Streamlit/Flask app (NUEVA)
└── components/                 ← Componentes UI (NUEVA)

notebooks/
├── 01_EDA.ipynb
├── 02_Preprocesamiento.ipynb
├── 03_Modelos_Clasificacion.ipynb
├── 04_Modelos_Regresion.ipynb
└── 05_Interpretabilidad_XAI.ipynb

tests/
├── test_data.py
├── test_preprocessing.py
├── test_models.py
└── test_evaluation.py
```

---

## 🎯 Cómo Funciona

### 1️⃣ Notebooks (Desarrollo)

```python
# notebooks/01_EDA.ipynb
from src.data.loader import load_and_clean_dataset
from src.visualization.eda import plot_distributions
from src.data.cleaner import convert_decimals

df = load_and_clean_dataset('data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv')
df = convert_decimals(df)
plot_distributions(df, ['MODALIDAD', 'PROMEDIO EDAD PROGRAMA'])
```

### 2️⃣ CLI (main.py)

```bash
# Ejecutar pipeline completo
python main.py --mode full --input data/raw/ --output data/processed/

# Entrenar modelos
python main.py --mode train --model-type classification

# Hacer predicciones
python main.py --mode predict --model classification --data data/test/

# Generar reporte
python main.py --mode report --format html
```

### 3️⃣ UI/Demo (Streamlit)

```bash
# Ejecutar interfaz interactiva
streamlit run ui/app.py
```

---

## 📁 Contenido de Archivos Principales

### `src/pipeline.py` (Orquestador)

```python
"""Pipeline central que coordina todo"""
from src.data.loader import load_and_clean_dataset
from src.preprocessing.transformers import apply_transformations
from src.models.classification import train_classifier
from src.models.regression import train_regressor
from src.evaluation.metrics import evaluate_models
from src.interpretation.xai import analyze_shap

class MLPipeline:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.X_train, self.X_test = None, None
        self.y_train_class, self.y_test_class = None, None
        self.y_train_reg, self.y_test_reg = None, None
        self.models = {}
    
    def load_and_prepare(self):
        """Cargar y preparar datos"""
        self.df = load_and_clean_dataset(self.config['data_path'])
        self.X_train, self.X_test, self.y_train_class, self.y_test_class, \
        self.y_train_reg, self.y_test_reg = apply_transformations(self.df)
        return self
    
    def train_all(self):
        """Entrenar todos los modelos"""
        self.models['classifier'] = train_classifier(self.X_train, self.y_train_class)
        self.models['regressor'] = train_regressor(self.X_train, self.y_train_reg)
        return self
    
    def evaluate_all(self):
        """Evaluar modelos"""
        results = {}
        results['classification'] = evaluate_models(
            self.models['classifier'], self.X_test, self.y_test_class, 'classification'
        )
        results['regression'] = evaluate_models(
            self.models['regressor'], self.X_test, self.y_test_reg, 'regression'
        )
        return results
    
    def interpret_models(self):
        """Generar explicabilidad"""
        return {
            'classifier_shap': analyze_shap(self.models['classifier'], self.X_test),
            'regressor_shap': analyze_shap(self.models['regressor'], self.X_test)
        }
    
    def run(self):
        """Ejecutar pipeline completo"""
        self.load_and_prepare()
        self.train_all()
        results = self.evaluate_all()
        interpretations = self.interpret_models()
        return results, interpretations
```

### `main.py` (CLI)

```python
"""Interfaz de línea de comandos"""
import argparse
from src.pipeline import MLPipeline
from src.config import load_config
import json

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline - Educación Superior Chile")
    
    # Modos de ejecución
    parser.add_argument('--mode', required=True, 
                       choices=['full', 'train', 'predict', 'report', 'eda'],
                       help='Modo de ejecución')
    
    # Configuración
    parser.add_argument('--config', default='config.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--input', default='data/raw/',
                       help='Directorio de entrada')
    parser.add_argument('--output', default='data/processed/',
                       help='Directorio de salida')
    
    # Modelos
    parser.add_argument('--model-type', choices=['classification', 'regression', 'both'],
                       default='both', help='Tipo de modelo a entrenar')
    
    # Reporte
    parser.add_argument('--format', choices=['html', 'pdf', 'txt'],
                       default='html', help='Formato del reporte')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    config['data_path'] = args.input
    config['output_path'] = args.output
    
    # Ejecutar según modo
    if args.mode == 'full':
        pipeline = MLPipeline(config)
        results, interpretations = pipeline.run()
        
        print("✓ Pipeline completado")
        print(json.dumps(results, indent=2))
        
    elif args.mode == 'train':
        pipeline = MLPipeline(config)
        pipeline.load_and_prepare()
        pipeline.train_all()
        print(f"✓ Modelos entrenados y guardados en {args.output}")
        
    elif args.mode == 'report':
        # Generar reporte
        from src.evaluation.reports import generate_report
        report = generate_report(format=args.format)
        print(f"✓ Reporte generado: {args.output}/reporte.{args.format}")
    
    elif args.mode == 'eda':
        # Ejecutar EDA
        from src.visualization.eda import run_full_eda
        run_full_eda(args.input, args.output)

if __name__ == '__main__':
    main()
```

### `ui/app.py` (Streamlit)

```python
"""Interfaz web interactiva con Streamlit"""
import streamlit as st
import pandas as pd
from src.pipeline import MLPipeline
from src.config import load_config

st.set_page_config(page_title="ML Demo - Educación Superior", layout="wide")

st.title("🎓 Modelado Predictivo - Educación Superior Chile")

with st.sidebar:
    st.header("⚙️ Configuración")
    mode = st.selectbox("Modo", ["EDA", "Entrenar", "Predecir", "Reportes"])
    model_type = st.selectbox("Tipo", ["Clasificación (Modalidad)", "Regresión (Edad)"])

if mode == "EDA":
    st.header("📊 Análisis Exploratorio")
    
    uploaded_file = st.file_uploader("Cargar CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        df.columns = df.columns.str.strip()
        
        st.write(f"Dataset: {df.shape[0]:,} registros × {df.shape[1]} columnas")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribución MODALIDAD")
            st.bar_chart(df['MODALIDAD'].value_counts())
        
        with col2:
            st.subheader("Estadísticas EDAD")
            st.write(df['PROMEDIO EDAD PROGRAMA'].describe())

elif mode == "Entrenar":
    st.header("🚀 Entrenamiento de Modelos")
    
    if st.button("Ejecutar Pipeline Completo"):
        with st.spinner("Entrenando modelos..."):
            config = load_config('config.yaml')
            pipeline = MLPipeline(config)
            results, _ = pipeline.run()
            
            st.success("✓ Entrenamiento completado")
            st.json(results)

elif mode == "Predecir":
    st.header("🔮 Hacer Predicciones")
    
    uploaded_file = st.file_uploader("Cargar datos para predecir", type="csv")
    
    if uploaded_file:
        df_test = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        
        if st.button("Predecir"):
            # Cargar modelo entrenado
            from src.models.classification import load_model
            model = load_model('models/classifier.pkl')
            
            predictions = model.predict(df_test)
            st.write("Predicciones:", predictions)

elif mode == "Reportes":
    st.header("📄 Reportes")
    
    report_type = st.radio("Tipo de reporte", 
                          ["Resumen EDA", "Resultados Modelos", "Interpretabilidad"])
    
    if st.button("Generar"):
        st.write(f"Generando reporte: {report_type}")
```

---

## 🚀 Cómo Ejecutar

### Opción 1: Jupyter Notebooks (Desarrollo)
```bash
cd notebooks
jupyter notebook 01_EDA.ipynb
```

### Opción 2: CLI (Automatización)
```bash
# Pipeline completo
python main.py --mode full

# Solo EDA
python main.py --mode eda --input data/raw/ --output data/processed/

# Entrenar modelos
python main.py --mode train --model-type classification

# Generar reporte
python main.py --mode report --format html
```

### Opción 3: UI Interactiva (Demo)
```bash
streamlit run ui/app.py
```

Luego abrir: `http://localhost:8501`

---

## ✅ Ventajas de Esta Arquitectura

| Aspecto | Beneficio |
|--------|----------|
| **Reutilización** | Código en `src/` se usa en notebooks, CLI y UI |
| **Testeable** | Fácil crear tests para cada módulo |
| **Escalable** | Agregar nuevas interfaces sin cambiar `src/` |
| **Producción** | Código limpio y documentado |
| **Demo** | UI para stakeholders |
| **Automatización** | CLI para pipelines automatizados |

---

## 📋 Migración Gradual

### Fase 1: Refactorizar `src/`
```
✓ Organizar módulos
✓ Crear pipeline.py
✓ Mantener main.py (mejorado)
✓ NO eliminar nada de src/
```

### Fase 2: Actualizar Notebooks
```
✓ Importar de src/
✓ Usar pipeline para reutilización
✓ Documentar bien
```

### Fase 3: Crear UI
```
✓ Streamlit app (ui/app.py)
✓ Usar pipeline.py como backend
✓ Componentes visuales
```

### Fase 4: Testing
```
✓ tests/test_data.py
✓ tests/test_models.py
✓ tests/test_pipeline.py
```

---

## 🎯 Conclusión

**NO ELIMINAR main.py** - Es la puerta de entrada para:
- ✅ Automatización (CI/CD)
- ✅ Producción (ML Ops)
- ✅ Demo (UI)
- ✅ Scripts reutilizables

**Mantener TODO en `src/`**, pero:
- Organizarlo mejor
- Hacerlo modular
- Usarlo desde múltiples interfaces

---

**¿Implemento esta arquitectura modular?** 🏗️
