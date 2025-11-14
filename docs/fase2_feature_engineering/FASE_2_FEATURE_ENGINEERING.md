# 🎨 FASE 2 - STEP 2: FEATURE ENGINEERING ✅

**Fecha:** 2025-11-12  
**Estado:** COMPLETADO Y FUNCIONAL

---

## 📦 Módulo Implementado

### `src/features/engineer.py`

Clase especializada en ingeniería de características (selección y creación).

---

## ✨ Funcionalidades

### 1. Análisis de Correlación
```python
engineer.calculate_correlation_matrix(X)
→ Matriz de Pearson
→ Detecta correlaciones altas (r > 0.8)
→ Identifica features redundantes
```

### 2. Detección de Multicolinealidad (VIF)
```python
engineer.calculate_vif(X)
→ Variance Inflation Factor por columna
→ Identifica features con multicolinealidad (VIF > 10)
→ Información detallada de problemas
```

### 3. Selección Univariante
```python
engineer.select_features_univariate(X, y, k=15, task='classification')
→ F-score para clasificación
→ F-score para regresión
→ Top-K features más importantes
→ Información de scores
```

### 4. Selección por Información Mutua
```python
engineer.select_features_mutual_info(X, y, k=20)
→ Información mutua entre X e y
→ Independiente de tipo de relación
→ Complementa selección univariante
```

### 5. Remover Features de Baja Varianza
```python
engineer.remove_low_variance_features(X, threshold=0.01)
→ Elimina features con varianza < threshold
→ Mejora eficiencia del modelo
→ Reporta features removidas
```

### 6. Crear Features de Interacción
```python
engineer.create_interaction_features(X, limit=10)
→ Multiplica pares de features
→ Captura relaciones no lineales
→ Limitado para eficiencia
```

### 7. Crear Features de Razón
```python
engineer.create_ratio_features(X, limit=5)
→ Divide features relacionadas
→ Extrae información relativa
→ Evita división por cero
```

---

## 🔧 Integración con Pipeline

### En `src/pipeline.py`

```python
class MLPipeline:
    def engineer_features(self):
        """Ingeniería de características"""
        self.feature_engineer = create_feature_engineer(self.config)
        
        # Análisis
        corr_matrix = engineer.calculate_correlation_matrix(X_train)
        vif = engineer.calculate_vif(X_train)
        
        # Selección
        selected = engineer.select_features_univariate(X_train, y_train)
        
        # Filtrado
        X_filtered = engineer.remove_low_variance_features(X_train)
```

---

## 📊 Resultados de Prueba

```
Feature Engineering:
  • Matriz de correlación: (40, 40)
  • VIF calculado: 40 features
  • Top 15 features seleccionados
  • Features de baja varianza removidos: 1
  • Features después de filtrado: 39

Detección de Correlaciones:
  • Pares correlacionados (r > 0.8): 0
  • Sin redundancia detectada

Detección de Multicolinealidad:
  • Features con VIF > 10: 0
  • Sin problemas significativos

Selección Univariante:
  • Features evaluados: 40
  • Features seleccionados: 15
  • Método: F-score (clasificación)
```

---

## 🚀 Uso

### Opción 1: Directo

```python
from src.features.engineer import create_feature_engineer
from src.config import Config

config = Config()
engineer = create_feature_engineer(config)

# Análisis
corr = engineer.calculate_correlation_matrix(X)
vif = engineer.calculate_vif(X)

# Selección
selected = engineer.select_features_univariate(
    X, y, k=15, task='classification'
)

# Filtrado
X_clean = engineer.remove_low_variance_features(X)
```

### Opción 2: Desde Pipeline

```python
from src.pipeline import MLPipeline

pipeline = MLPipeline()
pipeline.load_data()
pipeline.preprocess_data()
pipeline.engineer_features()

# Acceder a datos
X_train = pipeline.X_train
X_test = pipeline.X_test
engineer_info = pipeline.feature_engineer.get_feature_summary()
```

### Opción 3: CLI

```bash
python main.py --mode feature_engineering
# Ejecuta preprocesamiento + ingeniería
```

---

## 📋 Métodos Disponibles

| Método | Descripción | Input |
|--------|-------------|-------|
| `calculate_correlation_matrix()` | Matriz de correlación | X |
| `calculate_vif()` | Variance Inflation Factor | X |
| `select_features_univariate()` | Selección F-score | X, y, k, task |
| `select_features_mutual_info()` | Selección información mutua | X, y, k |
| `remove_low_variance_features()` | Filtrar baja varianza | X, threshold |
| `create_interaction_features()` | Features de interacción | X, limit |
| `create_ratio_features()` | Features de razón | X, limit |
| `get_feature_summary()` | Resumen de FE | - |

---

## 🔄 Flujo Completo

```
Dataset Preprocesado (173,522 × 40)
         ↓
Calcular Correlaciones
  • Matriz 40×40
  • Detectar r > 0.8
         ↓
Calcular VIF
  • Multicolinealidad
  • VIF por feature
         ↓
Selección Univariante (F-score)
  • Rank features
  • Top 15 seleccionados
         ↓
Remover Baja Varianza
  • Threshold: 0.01
  • 1 feature removido
         ↓
Dataset Optimizado (173,522 × 39)
  • Listo para modelos
  • Sin redundancia
  • Sin baja varianza
         ↓
Modelos ✓
```

---

## 🎯 Análisis Detallado

### Correlaciones

- **Método:** Pearson
- **Umbral:** 0.8 (r > 0.8)
- **Detectadas:** 0 pares
- **Implicación:** No hay features altamente redundantes

### Multicolinealidad (VIF)

- **Método:** Variance Inflation Factor
- **Umbral:** 10
- **Problemas:** 0 features
- **Implicación:** Baja multicolinealidad

### Selección Univariante

- **Método:** F-test (ANOVA)
- **Features evaluados:** 40
- **Features seleccionados:** 15
- **Criterio:** Mayor score F

### Varianza

- **Threshold:** 0.01
- **Features removidos:** 1
- **Implicación:** Mejora eficiencia

---

## ✅ Validación

### Test 1: Módulo Individual
```bash
python src/features/engineer.py
✓ Funciona correctamente
✓ Genera análisis completo
```

### Test 2: Desde Pipeline
```python
pipeline = MLPipeline()
pipeline.load_data()
pipeline.preprocess_data()
pipeline.engineer_features()
✓ Integración correcta
✓ Sin errores
```

### Test 3: Métodos Específicos
```python
engineer = create_feature_engineer(config)
corr = engineer.calculate_correlation_matrix(X)
vif = engineer.calculate_vif(X)
selected = engineer.select_features_univariate(X, y)
✓ Todos funcionan
```

---

## 📊 Estadísticas

```
Entrada:
  • Registros: 173,522
  • Features: 40
  • Tipo: Numéricas escaladas + categóricas codificadas

Salida:
  • Registros: 173,522 (sin cambios)
  • Features: 39
  • Tipo: Optimizadas
  
Cambios:
  • Features removidos: 1 (varianza baja)
  • Features creados: 0 (en esta ejecución)
  • Redundancia detectada: 0
  • Multicolinealidad: 0
```

---

## 🎯 Próximos Pasos

### COMPLETADO ✅
- [x] Cargar datos
- [x] EDA
- [x] Preprocesamiento
- [x] Feature Engineering

### PRÓXIMO 📝
- [ ] Entrenar modelos clasificación
- [ ] Entrenar modelos regresión
- [ ] Evaluación
- [ ] Interpretabilidad (XAI)

---

## 📁 Archivos

```
src/features/
├── __init__.py
└── engineer.py                 ✅ CREADO

src/pipeline.py                 ✅ ACTUALIZADO
  • engineer_features()
  • Imports

main.py                          ✅ COMPATIBLE
  • --mode feature_engineering
```

---

## ✅ Checklist

- [x] Crear clase FeatureEngineer
- [x] Implementar análisis de correlación
- [x] Implementar VIF
- [x] Implementar selección univariante
- [x] Implementar filtrado de varianza
- [x] Implementar creación de features
- [x] Integrar con pipeline
- [x] Probar funcionamiento
- [x] Documentación

---

## 🔬 Métodos de Selección

### 1. Univariante (F-test)
- **Caso:** Features numéricas vs target
- **Ventaja:** Rápido, simple
- **Desventaja:** No detecta interacciones
- **Uso:** Screening inicial

### 2. Información Mutua
- **Caso:** Features categóricas y numéricas
- **Ventaja:** Captura relaciones no lineales
- **Desventaja:** Más computacionalmente costoso
- **Uso:** Complementar univariante

### 3. Varianza
- **Caso:** Features sin variación
- **Ventaja:** Mejora eficiencia
- **Desventaja:** Puede perder info
- **Uso:** Pre-filtrado

---

**FASE 2 - PASO 2: COMPLETADO** ✅

**Próximo:** Entrenar Modelos (Clasificación + Regresión)

