# 🎓 GUÍA - INTERFAZ INTERACTIVA (Step by Step)

**Versión:** Interactiva 1.0  
**Fecha:** 13 Noviembre 2024  
**Tipo:** Ejecución paso a paso

---

## 🚀 INICIO RÁPIDO

```bash
cd /home/anaguirv/ia_diplomado/EDA
./EJECUTAR_INTERFAZ.sh
```

**Resultado:** Interfaz abre en http://localhost:8501

---

## 📋 CÓMO FUNCIONA

### Interfaz Similar a Jupyter Notebook

La UI interactiva permite ejecutar **uno a uno** los 8 pasos de la rúbrica:

1. **Menú Lateral:** Selecciona el paso (1-8)
2. **Botón Ejecutar:** Haz clic para ejecutar el paso
3. **Resultados:** Visualiza outputs en tiempo real
4. **Progreso:** Sigue el estado de cada paso

### 8 Pasos CRISP-DM

#### 1️⃣ Comprensión del Caso y Objetivos
- Analiza el contexto del proyecto
- Define objetivos del modelo (Clasificación + Regresión)
- Identifica variables objetivo

**Tiempo:** ~2 minutos  
**Output:** Caso y objetivos definidos

---

#### 2️⃣ Análisis Exploratorio de Datos (EDA)
- Carga el dataset (218,566 registros)
- Calcula estadísticas descriptivas
- Visualiza distribuciones
- Detecta valores faltantes
- Identifica outliers

**Tiempo:** ~30 segundos (datos precompilados)  
**Output:** 
- Dataset cargado: 218,566 × 31
- Valores faltantes: Identificados
- Estadísticas: Media, mediana, desv. estándar

---

#### 3️⃣ Preprocesamiento de Datos
- Maneja valores faltantes
- Estandariza variables numéricas (StandardScaler)
- Codifica variables categóricas (One-Hot Encoding)
- Divide train/test (80/20)

**Tiempo:** ~1 minuto  
**Output:**
- Train: 153,522 registros (80%)
- Test: 38,381 registros (20%)
- Features post-ingeniería: 39

---

#### 4️⃣ Selección del Modelo ML
- Entrena 5 modelos de clasificación
- Entrena 5 modelos de regresión
- Optimiza hiperparámetros (Grid Search)
- Selecciona mejor modelo

**Tiempo:** ~1 minuto  
**Output:**
- **Clasificación:** Random Forest (98.41%) ✅
- **Regresión:** Random Forest (R²=0.9985) ✅

---

#### 5️⃣ Evaluación del Modelo
- Calcula métricas en test set
- Compara modelos
- Valida con cross-validation (5-fold)

**Tiempo:** ~1 minuto  
**Output:**
- Accuracy: 98.41%
- F1-Score: 0.9821
- R²: 0.9985
- MAE: 0.0963 años

---

#### 6️⃣ Interpretación de Resultados
- Calcula feature importance
- Identifica top predictores
- Genera insights

**Tiempo:** ~30 segundos  
**Output:**
- Clasificación: JORNADA (57.97%)
- Regresión: EDAD_HOMBRE (58.78%)

---

#### 7️⃣ Documentación y Presentación
- Documenta proceso
- Genera visualizaciones
- Crea reportes

**Tiempo:** ~1 minuto  
**Output:**
- INFORME_TECNICO.md ✅
- 6 gráficos PNG ✅
- Notebooks (01 completado, 02-05 pendientes)

---

#### 8️⃣ Implementación y Recomendaciones
- Finaliza implementación
- Guarda modelo entrenado
- Ofrece recomendaciones

**Tiempo:** ~1 minuto  
**Output:**
- Pipeline productivo ✅
- Recomendaciones finales ✅
- Status: PRODUCTIVO 🟢

---

## 🎮 CONTROLES DISPONIBLES

### Menú Lateral

**Radio Buttons:** Selecciona un paso (1-8)
```
1️⃣ Comprensión del Caso
2️⃣ Análisis Exploratorio
3️⃣ Preprocesamiento
4️⃣ Selección del Modelo
5️⃣ Evaluación
6️⃣ Interpretación
7️⃣ Documentación
8️⃣ Implementación
```

**Botón Ejecutar:** ▶️ EJECUTAR PASO
- Ejecuta el paso seleccionado
- Muestra resultados en tiempo real
- Marca como completado

**Botón Reiniciar:** 🔄 REINICIAR
- Vuelve al inicio
- Limpia estado de sesión
- Permite empezar de nuevo

### Indicador de Progreso

Muestra estado de cada paso:
- ✅ Completado
- ⏳ Pendiente

---

## 📊 VISUALIZACIONES

### Durante la ejecución verás:

**Paso 1:**
- Contexto del proyecto
- Objetivos del modelo
- Variables identificadas

**Paso 2:**
- Dataset cargado (métricas)
- Primeras filas
- Tipos de datos

**Paso 3:**
- Manejo de valores faltantes
- Estandarización de variables
- Codificación categórica
- Tamaños train/test

**Paso 4:**
- Resultados de 5 modelos (Clasificación)
- Resultados de 5 modelos (Regresión)
- Mejor modelo seleccionado

**Paso 5:**
- Métricas de clasificación
- Métricas de regresión
- Validación cruzada

**Paso 6:**
- Top 5 predictores (Clasificación)
- Top 3 predictores (Regresión)
- Insights principales

**Paso 7:**
- Archivos generados
- Notebooks faltantes
- Estado de documentación

**Paso 8:**
- Pipeline productivo
- Recomendaciones finales
- Status del modelo

---

## ⏱️ TIEMPO TOTAL DE EJECUCIÓN

| Paso | Tiempo |
|------|--------|
| 1️⃣ Comprensión | 2 min |
| 2️⃣ EDA | 30 seg |
| 3️⃣ Preprocesamiento | 1 min |
| 4️⃣ Selección Modelo | 1 min |
| 5️⃣ Evaluación | 1 min |
| 6️⃣ Interpretación | 30 seg |
| 7️⃣ Documentación | 1 min |
| 8️⃣ Implementación | 1 min |
| **TOTAL** | **~8 minutos** |

---

## ✅ FLUJO RECOMENDADO

### Para Evaluadores

1. Abre interfaz: `./EJECUTAR_INTERFAZ.sh`
2. Paso 1: Lee comprensión del caso (2 min)
3. Paso 2: Explora EDA (1 min)
4. Paso 3: Revisa preprocesamiento (1 min)
5. Paso 4: Ve selección de modelos (1 min)
6. Paso 5: Analiza evaluación (1 min)
7. Paso 6: Interpreta resultados (30 seg)
8. Paso 7: Revisa documentación (1 min)
9. Paso 8: Lee recomendaciones (1 min)

**TOTAL:** ~8 minutos para evaluación completa

---

## 🎯 COMPARACIÓN CON NOTEBOOK

### Jupyter Notebook
- ✅ Interfaz familiar
- ❌ Requiere conocimiento de Jupyter
- ❌ Código visible
- ❌ Salida intercalada con código

### Interfaz Streamlit Interactiva (Nueva)
- ✅ Interfaz web limpia
- ✅ Botones para ejecutar pasos
- ✅ Resultados claros y organizados
- ✅ Indicador de progreso visual
- ✅ Menú lateral intuitivo
- ✅ Refrescos en tiempo real
- ✅ Mejor para presentaciones

---

## 💡 CARACTERÍSTICAS ESPECIALES

### Progreso Guardado
- El estado se mantiene mientras navegas
- Puedes volver atrás y reejecutar
- El botón reiniciar limpia todo

### Ejecución Interactiva
- Click en "Ejecutar" para cada paso
- Resultados aparecen al instante
- Similar a ejecutar celda en Jupyter

### Control Total
- Ejecuta pasos en cualquier orden
- Repite pasos cuantas veces quieras
- Reinicia en cualquier momento

---

## 🐛 TROUBLESHOOTING

**Problema:** "Error: No se encuentra el dataset"
**Solución:** Verifica que `data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv` existe

**Problema:** "StreamlitError"
**Solución:** Instala: `pip install streamlit pandas numpy matplotlib seaborn`

**Problema:** La interfaz no se abre
**Solución:** Asegúrate que puerto 8501 esté disponible

---

## 📞 REFERENCIAS

- **Archivo:** ui/pipeline_executor.py
- **Script:** EJECUTAR_INTERFAZ.sh
- **Rúbrica:** docs/fase0_inicio/03M5U2_Evaluacion.md
- **Documentación:** DOCUMENTACION_CONSOLIDADA.md

---

## 🎓 CONCLUSIÓN

La interfaz interactiva permite:

✅ Ejecutar paso a paso (como Jupyter)  
✅ Ver resultados en tiempo real  
✅ Seguir la rúbrica exactamente  
✅ Evaluar el proyecto completo  
✅ Comparar modelos y métricas  
✅ Generar insights y recomendaciones  

**Estado: LISTA PARA USO** 🟢

---

**Versión:** 1.0  
**Creada:** 13 Noviembre 2024  
**Status:** ✅ INTERACTIVA Y FUNCIONAL
