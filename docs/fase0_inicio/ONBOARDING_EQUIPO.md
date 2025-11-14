# 🚀 Onboarding - Proyecto ML: Modelado Predictivo Educación Superior Chile

**Fecha de creación:** 11 de Noviembre, 2025  
**Estado del Proyecto:** Iniciado - Fase de Implementación  
**Responsable:** Equipo de Desarrollo ML

---

## 📌 Contexto del Proyecto

Este documento guía a nuevos miembros del equipo para iniciar la implementación del proyecto de Machine Learning de forma consistente con las sesiones anteriores.

### Objetivos del Proyecto
1. **Clasificación (Tarea 1):** Predecir **MODALIDAD** de programas (Presencial vs No Presencial)
2. **Regresión (Tarea 2):** Predecir **PROMEDIO EDAD PROGRAMA** (edad continua)

### Dataset
- **Ubicación:** `data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv`
- **Tamaño:** 218,566 registros
- **Variables Objetivo:** MODALIDAD, PROMEDIO EDAD PROGRAMA
- **Variables Explicativas:** Área CINE, región, institución, jornada, duración, comuna, nivel institucional, año, etc.

### Requisitos del Proyecto
- **Documento oficial:** `docs/requerimientos_proyecto.md`
- **Especificación completa:** `03M5U2_Evaluacion.md`
- **Metodología:** CRISP-DM con 4 fases (Ideación, Preparación, Entrenamiento, Evaluación)

---

## 🎯 Primer Paso: Prompt Inicial para Copilot/IA

Cuando inices una sesión con GitHub Copilot o tu asistente de IA favorito, **copia y pega exactamente este prompt:**

```
CONTEXTO DEL PROYECTO:
====================

Proyecto: Modelado Predictivo para Optimización de Educación Superior en Chile
Ubicación: /home/anaguirv/ia_diplomado/EDA/
Responsable: Equipo de Desarrollo ML

TAREAS ML PROPUESTAS:
1. Clasificación Binaria: Predecir MODALIDAD (Presencial vs No Presencial)
2. Regresión: Predecir PROMEDIO EDAD PROGRAMA (valor continuo)

DATASET:
- Ruta: data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv
- Registros: 218,566
- Período: 2007-2024
- Separador: punto y coma (;)
- Encoding: UTF-8 con caracteres especiales españoles

VARIABLES OBJETIVO:
- Y1: MODALIDAD (Presencial / No Presencial)
- Y2: PROMEDIO EDAD PROGRAMA

VARIABLES EXPLICATIVAS PRINCIPALES:
- Área CINE (CINE-F_97 ÁREA, CINE-F_13 ÁREA)
- REGIÓN, PROVINCIA, COMUNA
- CLASIFICACIÓN INSTITUCIÓN (NIVEL 1, 2, 3)
- JORNADA (Diurna, Vespertina, A Distancia)
- MODALIDAD (Presencial, No Presencial)
- DURACIÓN ESTUDIO CARRERA
- TIPO DE PLAN DE LA CARRERA
- AÑO (2007-2024)

ESTRUCTURA DEL PROYECTO:
- data/raw/          → Dataset original
- data/processed/    → Datos procesados (crear si no existe)
- notebooks/         → Notebooks de análisis y modelos
- src/               → Código Python reutilizable
- scripts/           → Scripts de utilidad
- tests/             → Tests unitarios
- venv/              → Virtual environment

REQUISITOS:
- Python 3.9+
- pandas, numpy, scikit-learn
- tensorflow/keras para deep learning
- matplotlib, seaborn para visualizaciones
- SHAP para interpretabilidad
- requirements.txt debe mantenerse actualizado

METODOLOGÍA:
Seguir CRISP-DM con 4 fases:
1. IDEACIÓN: Definir métricas, baseline, estrategia
2. PREPARACIÓN: EDA, preprocesamiento, feature engineering
3. ENTRENAMIENTO: Seleccionar algoritmos, tuning hiperparámetros
4. EVALUACIÓN: Evaluación en test set, interpretabilidad (XAI)

ENTREGABLES ESPERADOS:
1. 01_EDA.ipynb - Análisis exploratorio completo
2. 02_Preprocesamiento.ipynb - Limpieza y normalización
3. 03_Modelos_Clasificacion.ipynb - Modelos para MODALIDAD
4. 04_Modelos_Regresion.ipynb - Modelos para EDAD
5. 05_Interpretabilidad_XAI.ipynb - SHAP, Feature Importance
6. INFORME_TECNICO.md - Documentación final con resultados

MÉTRICAS DE ÉXITO:
- Clasificación: AUC-PR, F1-Score, Matriz de Confusión
- Regresión: MAE, RMSE
- XAI: Permutation Importance, SHAP values

ETAPAS COMPLETADAS:
✓ Limpieza de directorio (eliminadas docs innecesarias)
✓ Definición de caso y dataset
✓ Preparación de estructura del proyecto


INSTRUCCIONES PARA TI (Asistente IA):
1. Mantén el contexto de todas las sesiones anteriores
2. Consulta docs/requerimientos_proyecto.md para decisiones técnicas
3. Verifica 03M5U2_Evaluacion.md para requisitos de evaluación
4. No realices cambios sin validar contra la estructura definida
5. Documenta claramente cada decisión de preprocesamiento
6. Incluye validación de data leakage en cada etapa
7. Genera logs y reportes interpretables para stakeholders

```

---

## 📂 Estructura de Directorios Actualizada

```
EDA/
├── data/
│   ├── raw/
│   │   └── TITULADO_2007-2024_web_19_05_2025_E.csv (218K registros)
│   └── processed/
│       ├── train_X.csv
│       ├── train_y_modalidad.csv
│       ├── train_y_edad.csv
│       ├── test_X.csv
│       ├── test_y_modalidad.csv
│       └── test_y_edad.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 03_Modelos_Clasificacion.ipynb
│   ├── 04_Modelos_Regresion.ipynb
│   └── 05_Interpretabilidad_XAI.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── xai.py
├── scripts/
│   ├── train_pipeline.py
│   ├── predict.py
│   └── generate_report.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── docs/
│   └── requerimientos_proyecto.md
├── README.md
├── 03M5U2_Evaluacion.md
├── ONBOARDING_EQUIPO.md (este archivo)
├── INFORME_TECNICO.md (se crea al finalizar)
├── requirements.txt
├── .gitignore
└── venv/
```

---

## ✅ Checklist de Inicio Rápido

Cuando un nuevo colega inicia sesión:

- [ ] 1. Leer este archivo (ONBOARDING_EQUIPO.md) completamente
- [ ] 2. Revisar `docs/requerimientos_proyecto.md` para contexto de negocio
- [ ] 3. Revisar `03M5U2_Evaluacion.md` para criterios de evaluación
- [ ] 4. Verificar que el dataset existe en `data/raw/`
- [ ] 5. Copiar y pegar el **Prompt Inicial** completo a tu sesión con IA
- [ ] 6. Validar que el virtual environment esté activo: `source venv/bin/activate`
- [ ] 7. Instalar/actualizar dependencias: `pip install -r requirements.txt`
- [ ] 8. Explorar estructura actual: `ls -la` y `tree` (opcional)
- [ ] 9. Revisar README.md para instrucciones adicionales

---

## 🔑 Decisiones Técnicas Clave

### 1. Split Train/Test
- **Ratio:** 80/20 (por definir en EDA si hay temporal dimension)
- **Estrategia:** Random split (sin data leakage temporal)
- **Validación:** 10% del training set para early stopping

### 2. Preprocesamiento
- **Valores faltantes:** Analizar por columna (imputación vs eliminación)
- **Outliers:** Detección con IQR y análisis visual
- **Escalado:** StandardScaler para modelos sensibles a escala
- **Encoding:** One-Hot Encoding para variables categóricas

### 3. Modelos Candidatos
**Clasificación (MODALIDAD):**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)
- Red Neuronal (Dense + Dropout + L2)

**Regresión (EDAD):**
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- Red Neuronal (Dense + Dropout + L2)

### 4. Regularización
- **Dropout:** 20-30% en capas intermedias
- **Weight Decay (L2):** 0.001-0.01
- **Early Stopping:** paciencia de 10-20 épocas

### 5. Métricas Principales
**Clasificación:**
- AUC-PR (área bajo la curva precisión-recall)
- F1-Score
- Matriz de Confusión
- Umbral operativo optimizado

**Regresión:**
- MAE (Mean Absolute Error) - primario
- RMSE (Root Mean Squared Error) - secundario
- R² Score

---

## 🔗 Referencias Útiles

| Documento | Ubicación | Propósito |
|-----------|-----------|----------|
| Especificación Técnica | `docs/requerimientos_proyecto.md` | Contexto de negocio, metodología, métricas |
| Rúbrica de Evaluación | `03M5U2_Evaluacion.md` | Criterios de evaluación y puntuación |
| README | `README.md` | Instrucciones generales del proyecto |
| Este Onboarding | `ONBOARDING_EQUIPO.md` | Guía de inicio rápido para colegas |

---

## 💬 Comunicación del Equipo

### Convenciones de Código
- Nombres en Python: `snake_case` para funciones/variables
- Docstrings: Google Style
- Type hints: Obligatorios en funciones
- Comentarios: Solo para lógica compleja

### Versionado
- Rama `main`: código probado y documentado
- Ramas de feature: `feature/eda`, `feature/preprocessing`, etc.
- Commits: descriptivos, e.g., `feat: agregar limpieza de valores faltantes`

### Documentación
- Cada notebook debe tener celdas markdown explicativas
- Cada función debe tener docstring
- Cambios significativos se documentan en un `CHANGELOG.md`

---

## 🚨 Alertas de Data Leakage

**CRÍTICO:** Evitar a toda costa:
- ❌ Usar información del test set en preprocesamiento
- ❌ Calcular estadísticas de scaling con todo el dataset
- ❌ Crear features basadas en datos futuros
- ❌ Balancear clases antes de split train/test
- ✅ HACER: Fit escaladores/imputadores SOLO en train set

---

## 📞 Contacto y Escaladas

Si encuentras dudas:
1. Consulta `docs/requerimientos_proyecto.md`
2. Revisa notebooks anteriores completados
3. Documenta el problema en un comentario con contexto
4. Escala al responsable del proyecto

---

## 📝 Plantilla de Sesión Nueva

Cuando inicia una nueva sesión, usa este template:

```markdown
# Sesión: [Nombre/Objetivo]
**Fecha:** [YYYY-MM-DD]
**Responsable:** [Nombre]
**Tarea:** [Descripción breve]

## Contexto
[Referencia a documentos/decisiones previas]

## Objetivos de esta Sesión
1. [Objetivo 1]
2. [Objetivo 2]
3. [Objetivo 3]

## Entregables
- [ ] [Entregable 1]
- [ ] [Entregable 2]

## Notas Técnicas
[Decisiones, algoritmos, configuraciones]

## Resultados
[Resumen de lo logrado]
```

---

## ✨ Éxito Esperado

Al finalizar el proyecto deberás tener:

✅ **01_EDA.ipynb** - Dataset completamente explorado  
✅ **02_Preprocesamiento.ipynb** - Datos listos para modelado  
✅ **03_Modelos_Clasificacion.ipynb** - Modelos evaluados para MODALIDAD  
✅ **04_Modelos_Regresion.ipynb** - Modelos evaluados para EDAD  
✅ **05_Interpretabilidad_XAI.ipynb** - Insights accionables  
✅ **INFORME_TECNICO.md** - Documentación ejecutiva  
✅ **src/** - Código reutilizable y testeado  
✅ **tests/** - Suite de tests completa  

---

**¡Bienvenido al equipo! 🎉 Adelante con la implementación.**

Documento creado: 11/11/2025 - v1.0
