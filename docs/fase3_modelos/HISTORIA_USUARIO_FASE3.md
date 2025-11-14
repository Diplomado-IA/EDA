# Historia de Usuario: Fase 3 - Modelado Predictivo

## Contexto
Como **Equipo de Data Science**, necesitamos implementar modelos predictivos para clasificar el estado de titulación de estudiantes, basándonos en las variables ingenierizadas en la Fase 2.

**Estado actual**: Datos preprocesados y features construidas  
**Objetivo**: Modelos entrenados, evaluados y listos para producción

---

## 📋 Historia de Usuario

### ID: FASE3-001
**Título**: Desarrollo de modelos predictivos para clasificación de titulación

**Como** científico de datos  
**Quiero** entrenar, evaluar y seleccionar los mejores modelos predictivos  
**Para** poder hacer predicciones precisas sobre el estado de titulación de estudiantes

**Contexto**: El dataset ya está preprocesado (Fase 1-2), features ingenierizadas, normalizado y dividido en train/test.

---

## ✅ Criterios de Aceptación (Gherkin)

### Escenario 1: Entrenamiento de modelos base
```gherkin
Escenario: Entrenar múltiples algoritmos clasificadores
  Dado que tengo el dataset preprocesado en "data/processed/final_dataset.csv"
  Y tengo features seleccionadas documentadas en "src/config/features_config.yml"
  
  Cuando entreno los siguientes modelos:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - SVM
    - Neural Network
  
  Entonces cada modelo debe:
    ✓ Converger sin errores
    ✓ Generar métricas base (Accuracy, Precision, Recall, F1)
    ✓ Ser guardado en "models/trained/[model_name]_v1.pkl"
    ✓ Registrar sus hiperparámetros en "models/metadata/training_log.json"
```

### Escenario 2: Evaluación y validación cruzada
```gherkin
Escenario: Validar modelos con K-Fold Cross-Validation
  Dado que tengo 5 modelos entrenados
  Y utilizo 5-Fold Cross-Validation
  
  Cuando evalúo cada modelo
  
  Entonces debo obtener:
    ✓ Scores de CV con desviación estándar < 5%
    ✓ Matriz de confusión por clase
    ✓ Curva ROC-AUC
    ✓ Reporte de clasificación completo
    ✓ Tabla comparativa de modelos en "outputs/model_comparison.html"
```

### Escenario 3: Selección del mejor modelo
```gherkin
Escenario: Identificar modelo óptimo
  Dado que tengo métricas de evaluación de 5 modelos
  Y los criterios de selección son: F1-Score (60%), Recall (30%), Latencia (10%)
  
  Cuando aplico ponderación de criterios
  
  Entonces:
    ✓ Modelo ganador tiene F1 > 0.75
    ✓ Recall > 0.70 (minimizar falsos negativos)
    ✓ Latencia inferencia < 100ms
    ✓ Se exporta como "models/production/best_model_v1.pkl"
```

### Escenario 4: Análisis de importancia de features
```gherkin
Escenario: Entender contribución de features
  Dado que tengo el modelo seleccionado
  
  Cuando calculo importancia de features
  
  Entonces:
    ✓ Top 15 features identificados
    ✓ Gráfico SHAP exportado a "outputs/feature_importance.png"
    ✓ Gráfico Permutation Importance generado
    ✓ Análisis guardado en "docs/fase3_modelos/ANALISIS_FEATURES.md"
```

---

## 🔧 Tareas de Programación

### Sprint 1: Setup y Modelos Base

**TAREA-3.1**: Crear estructura de training
```python
# src/models/training.py
- Función: load_data_split(test_size=0.2, val_size=0.1)
- Función: get_base_models() -> dict
- Función: train_model(model, X_train, y_train) -> trained_model
- Logging: Registrar tiempos de entrenamiento y recursos
```

**TAREA-3.2**: Implementar Logistic Regression + Random Forest
```python
# src/models/classifiers.py
- LR: parametrización (C, solver, max_iter)
- RF: parametrización (n_estimators, max_depth, min_samples)
- Grid Search básico para cada uno
- Exportación de modelos entrenados
```

**TAREA-3.3**: Implementar Gradient Boosting + SVM
```python
# src/models/advanced_models.py
- GB: XGBoost o LightGBM con tuning
- SVM: Kernel selection + C parameter
- Validación cruzada K=5
```

### Sprint 2: Evaluación y Comparación

**TAREA-3.4**: Módulo de evaluación
```python
# src/models/evaluation.py
- Función: evaluate_model(y_true, y_pred) -> MetricsDict
- Función: cross_validate_models(models, X, y, k=5)
- Función: generate_confusion_matrix(y_true, y_pred)
- Función: plot_roc_curves(models_results)
```

**TAREA-3.5**: Dashboard de comparación
```python
# notebooks/03_MODEL_EVALUATION.ipynb
- Tabla comparativa con métricas normalizadas
- Gráficos de rendimiento lado a lado
- Matriz de correlación de predicciones
- Exportar resumen en HTML interactivo
```

**TAREA-3.6**: Análisis de importancia
```python
# src/models/interpretability.py
- Función: calculate_feature_importance(model, X)
- Función: plot_shap_values(model, X)
- Función: permutation_importance(model, X_test, y_test)
```

### Sprint 3: Selección y Producción

**TAREA-3.7**: Mecanismo de selección
```python
# src/models/model_selection.py
- Función: weighted_score(metrics, weights)
- Función: select_best_model(models_results, criteria)
- Exportar campeón a models/production/
```

**TAREA-3.8**: Validación en test set
```python
# notebooks/04_FINAL_VALIDATION.ipynb
- Predicciones en test set virgen
- Reporte final de performance
- Comparación train vs test (detectar overfitting)
- Umbral de decisión óptimo
```

**TAREA-3.9**: Documentación de modelos
```python
# docs/fase3_modelos/MODELOS_FINALES.md
- Especificaciones técnicas de cada modelo
- Hiperparámetros óptimos
- Performance metrics finales
- Recomendaciones de uso
```

---

## 📊 Entregables Esperados

```
outputs/
├── model_comparison.html          # Tabla interactiva
├── feature_importance.png         # Top 15 features
├── confusion_matrices.png         # 2x3 subplot
├── roc_curves.png                 # Todas las curvas
└── model_performance_report.pdf   # Resumen ejecutivo

models/
├── production/
│   └── best_model_v1.pkl         # Modelo ganador
├── trained/
│   ├── lr_v1.pkl
│   ├── rf_v1.pkl
│   ├── gb_v1.pkl
│   ├── svm_v1.pkl
│   └── nn_v1.pkl
└── metadata/
    ├── training_log.json
    └── model_cards/
        └── best_model_v1_card.md

notebooks/
├── 03_MODEL_EVALUATION.ipynb      # Comparación de modelos
└── 04_FINAL_VALIDATION.ipynb      # Validación final

docs/fase3_modelos/
├── MODELOS_FINALES.md             # Especificaciones
└── ANALISIS_FEATURES.md           # Importancia de features
```

---

## 🎯 Criterios de Éxito

| Criterio | Umbral | Prioridad |
|----------|--------|-----------|
| F1-Score (Test Set) | > 0.75 | 🔴 Alta |
| Recall (clase minoritaria) | > 0.70 | 🔴 Alta |
| Overfitting (|train_f1 - test_f1|) | < 0.05 | 🟡 Media |
| Latencia predicción | < 100ms | 🟢 Baja |
| Reproducibilidad (seed fijo) | Determinístico | 🟡 Media |

---

## 📝 Notas

- **Balanceo de clases**: Usar SMOTE si hay desbalance significativo
- **Feature scaling**: Ya aplicado en Fase 2, verificar en training
- **Baseline**: Iniciar con dummy classifier para comparación
- **Hyperparameter tuning**: Usar Optuna o GridSearchCV
- **Reproducibilidad**: Fijar random_state=42 en todos los modelos

---

## 🔗 Dependencias

- ✅ Fase 1: EDA completado
- ✅ Fase 2: Feature Engineering completado
- 📦 Requerimientos: sklearn, xgboost, tensorflow, shap, optuna
