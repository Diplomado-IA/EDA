# 🤝 ONBOARDING - Contexto para Nuevos Miembros

## Para Copilot/IA: Archivo de Instrucciones del Proyecto

Este documento define el **contexto y las instrucciones** que debe entender cualquier agente (humano o IA) que trabaje en este proyecto.

---

## 📌 Misión del Proyecto

**Objetivo General:** Construir un modelo predictivo que clasifique el estado de titulación de estudiantes de educación superior en Chile (2007-2024).

**Dataset:** `data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv` (173,522 registros, 40 columnas)

---

## 🏗️ Arquitectura Modular

El proyecto utiliza una **arquitectura modular de 5 capas**:

```
1. RAW DATA → 2. PREPROCESSING → 3. FEATURE ENGINEERING → 4. MODELING → 5. DEPLOYMENT
   Fase 1       Fase 2.1          Fase 2.2               Fase 3        Fase 4
```

Cada fase tiene su propia documentación en `docs/faseX_nombre/`.

---

## 📁 Estructura de Directorios (SOLO LO NECESARIO)

```
/
├── data/raw/                    # Datos originales (NO editar)
│   └── TITULADO_2007-2024...csv
├── data/processed/              # Datos limpios y features (Fases 2-3)
├── notebooks/                   # Jupyter notebooks (Ejecución interactiva)
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model_Training.ipynb
├── src/                         # Código reutilizable (módulos)
│   ├── config.py                # Configuración global
│   ├── pipeline.py              # Orquestador principal
│   ├── preprocessing/           # Limpieza de datos
│   ├── features/                # Ingeniería de features
│   └── models/                  # Entrenamiento y evaluación
├── models/                      # Modelos entrenados
│   ├── production/              # Modelo ganador
│   ├── trained/                 # Todos los modelos
│   └── metadata/                # Logs y specifications
├── outputs/                     # Gráficos, reportes, visualizaciones
├── ui/                          # Dashboard Streamlit
│   └── app.py                   # Interfaz de usuario
├── docs/                        # Documentación detallada (LEER PRIMERO)
│   ├── fase0_inicio/            # Requerimientos y onboarding
│   ├── fase1_eda/               # Análisis exploratorio
│   ├── fase2_preprocesamiento/  # Limpieza
│   ├── fase2_feature_engineering/ # Features
│   ├── fase3_modelos/           # Modelado
│   ├── arquitectura/            # Especificaciones técnicas
│   └── integracion/             # Pipeline completo
├── requirements.txt             # Dependencias Python
├── README.md                    # Este archivo
└── .gitignore                   # Archivos ignorados en git
```

**❌ NO incluir en raíz:**
- main.py (obsoleto)
- DOCUMENTACION.md (contenido en docs/)
- ESTRUCTURA.md (actualizado en docs/)
- scripts/ (si está vacío)

---

## 🚀 Inicio Rápido (5 minutos)

### Paso 1: Clonar y configurar
```bash
cd /home/anaguirv/ia_diplomado/EDA
source venv/bin/activate  # O crear: python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Paso 2: Verificar setup
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv')
print(f'✓ Datos cargados: {df.shape}')
"
```

### Paso 3: Ver documentación de tu fase
```bash
# Reemplaza X con tu fase (0, 1, 2, 3, etc.)
cat docs/faseX_nombre/INDICE.md
```

---

## 📖 Fases del Proyecto

### ✅ FASE 0: INICIO
**Responsables:** Product Manager + Tech Lead  
**Entrega:** Requerimientos y setup  
**Documentación:** `docs/fase0_inicio/`  
**Status:** ✅ COMPLETADO

**Qué hacer si necesitas info:**
```bash
cat docs/fase0_inicio/QUICK_START.md              # Inicio rápido
cat docs/requerimientos_proyecto.md              # Qué se pide
cat docs/fase0_inicio/03M5U2_Evaluacion.md       # Evaluación del curso
```

---

### ✅ FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
**Notebook:** `notebooks/01_EDA.ipynb`  
**Documentación:** `docs/fase1_eda/INDICE.md`  
**Status:** ✅ COMPLETADO

**Qué salió:**
- 40 variables analizadas (tipos de datos, nulos, distribuciones)
- Gráficos de distribución en `outputs/`
- Detección de outliers y anomalías
- Correlaciones entre variables

**Si necesitas regenerar:**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

### ✅ FASE 2.1: PREPROCESAMIENTO
**Notebook:** `notebooks/02_Preprocesamiento.ipynb`  
**Documentación:** `docs/fase2_preprocesamiento/INDICE.md`  
**Status:** ✅ COMPLETADO

**Transformaciones aplicadas:**
- Limpieza de valores nulos
- Manejo de outliers
- Normalización/Escalado
- Encoding de variables categóricas
- Balanceo de clases (si aplica)

**Salida:** `data/processed/preprocessed_data.csv`

---

### ✅ FASE 2.2: FEATURE ENGINEERING
**Notebook:** `notebooks/03_Feature_Engineering.ipynb`  
**Documentación:** `docs/fase2_feature_engineering/INDICE.md`  
**Status:** ✅ COMPLETADO

**Transformaciones aplicadas:**
- Creación de variables derivadas (ratios, interacciones)
- Selección de features relevantes
- Reducción dimensional (si es necesario)
- Validación de features

**Salida:** `data/processed/final_dataset.csv`  
**Módulo:** `src/features/engineer.py`

---

### 🔄 FASE 3: MODELADO PREDICTIVO
**Notebooks:** `03_MODEL_EVALUATION.ipynb` → `04_FINAL_VALIDATION.ipynb`  
**Documentación:** `docs/fase3_modelos/HISTORIA_USUARIO_FASE3.md`  
**Status:** 🔄 EN PROGRESO

**Qué hacer:**

1. **Lee la especificación completa:**
```bash
cat docs/fase3_modelos/HISTORIA_USUARIO_FASE3.md
```

2. **Estructura:**
   - Sprint 1: Entrenar 5 modelos base (LR, RF, GB, SVM, NN)
   - Sprint 2: Evaluar con K-Fold CV, generar reportes comparativos
   - Sprint 3: Seleccionar mejor modelo, validación final

3. **Criterios de Éxito:**
   - F1-Score Test > 0.75
   - Recall > 0.70
   - Documentación completa en `docs/fase3_modelos/MODELOS_FINALES.md`

4. **Crear notebooks:**
```bash
jupyter notebook notebooks/03_MODEL_EVALUATION.ipynb
jupyter notebook notebooks/04_FINAL_VALIDATION.ipynb
```

---

## 🔧 Instrucciones Técnicas Específicas

### Cómo ejecutar un notebook
```bash
# Opción 1: Jupyter interactivo
jupyter notebook notebooks/01_EDA.ipynb

# Opción 2: Terminal (para testing automático)
jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb
```

### Cómo verificar que todo funciona
```bash
# Test del pipeline completo
python -c "
from src.pipeline import MLPipeline
pipeline = MLPipeline()
print('✓ Pipeline OK')
"

# Test de datos
python -c "
import pandas as pd
from src.preprocessing.preprocessor import Preprocessor
df = pd.read_csv('data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv')
prep = Preprocessor()
clean_df = prep.fit_transform(df)
print(f'✓ Preprocesamiento OK: {clean_df.shape}')
"
```

### Cómo usar la UI
```bash
streamlit run ui/app.py
# Abre en http://localhost:8501
```

---

## 📊 Configuración Global

Todos los parámetros globales están en `src/config.py`:

```python
# Rutas de datos
DATA_RAW = 'data/raw/'
DATA_PROCESSED = 'data/processed/'
MODELS_PATH = 'models/production/'

# Parámetros de modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Modelos a entrenar
MODELS_TO_TRAIN = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVM', 'NeuralNetwork']

# Criterios de selección
SELECTION_CRITERIA = {
    'f1_score': 0.60,
    'recall': 0.30,
    'latency': 0.10
}
```

Modifica aquí si necesitas cambiar comportamiento global.

---

## 🚨 Troubleshooting

### Error: ModuleNotFoundError: No module named 'src'
**Solución:**
```bash
cd /home/anaguirv/ia_diplomado/EDA  # Asegurate de estar en la raíz
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Error: FileNotFoundError: 'data/raw/TITULADO...'
**Solución:**
```bash
# Verifica que existe el archivo
ls -la data/raw/
# Si no existe, descárgalo del fuente de datos
```

### Error: 'PROMEDIO EDAD PROGRAMA' no existe
**Solución:**
- Verifica el nombre exacto de columnas: `df.columns`
- Algunos notebooks tienen espacios extra. Ajusta el nombre si es necesario

### Error: KeyError en visualizaciones
**Solución:**
```bash
# Regenera datos procesados
jupyter nbconvert --to notebook --execute notebooks/02_Preprocesamiento.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_Feature_Engineering.ipynb
```

---

## 📝 Checklist para Comenzar

- [ ] Clone/actualice el repo
- [ ] Active el venv: `source venv/bin/activate`
- [ ] Instale dependencias: `pip install -r requirements.txt`
- [ ] Lea `docs/requerimientos_proyecto.md`
- [ ] Identifique su fase de trabajo
- [ ] Lea INDICE.md de su fase
- [ ] Verifique setup: `python -c "import pandas; print('OK')"`
- [ ] Ejecute primer notebook de su fase

---

## 🔗 Referencias Rápidas

| Necesito... | Ir a... |
|------------|---------|
| Entender el proyecto | `docs/requerimientos_proyecto.md` |
| Especificación técnica | `docs/arquitectura/ARQUITECTURA_MODULAR.md` |
| Cómo ejecutar código | `docs/arquitectura/GUIA_EJECUCION_MODULAR.md` |
| Qué ya se hizo | `docs/integracion/VERIFICACION_README.md` |
| Configuración global | `src/config.py` |
| Modelos entrenados | `models/trained/` y `models/production/` |
| Gráficos y reportes | `outputs/` |

---

## 👥 Roles y Responsabilidades

- **Data Engineer:** Fase 2.1 (Preprocesamiento) - `src/preprocessing/`
- **Data Scientist:** Fase 1 + 2.2 + 3 (EDA, Features, Modelos)
- **ML Engineer:** Fase 3 (Productivización de modelos)
- **Frontend Dev:** `ui/app.py` (Dashboard Streamlit)

---

## 🎓 Filosofía del Proyecto

1. **Reproducibilidad:** Seed fijo (42), versiones pinned en requirements.txt
2. **Modularidad:** Cada fase separada, reutilizable
3. **Documentación:** Inline comments para código complejo, docstrings en funciones
4. **Testing:** Validaciones en cada notebook antes de exportar
5. **Trazabilidad:** Logs de entrenamiento, metadata de modelos

---

## 📞 Preguntas Frecuentes

**P: ¿Por qué hay notebooks si tenemos módulos en src/?**  
A: Los notebooks son para exploración e iteración rápida. El código final se refactoriza en src/ para reutilización.

**P: ¿Cómo se ejecuta sin main.py?**  
A: Se ejecuta desde notebooks (interactivo) o `streamlit run ui/app.py` (producción).

**P: ¿Dónde agrego mis propias features?**  
A: En `src/features/engineer.py`, función `create_new_features()`.

**P: ¿Cómo guardo mi modelo?**  
A: Automático en `models/trained/` desde `src/models/trainer.py`.

---

**Última actualización:** 2025-11-12  
**Versión:** 1.0 - Contexto Completo para Nuevos Miembros
