# 🤖 FASE 3: ENTRENAMIENTO DE MODELOS

## Índice de Documentación

### Referencia: `UI_MEJORADA.md`
Documentación de la interfaz Streamlit y visualización de resultados.

### Componentes:
- **Módulo**: `src/models/trainer.py`
- **Notebook**: `notebooks/04_Model_Training.ipynb`
- **UI**: `ui/app.py` (Streamlit)

### Objetivos:
1. ✅ Entrenar modelos de clasificación
2. ✅ Entrenar modelos de regresión
3. ✅ Evaluar rendimiento
4. ✅ Generar métricas
5. ✅ Visualizar resultados en UI

### Modelos a Entrenar:

#### Clasificación:
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Neural Networks

#### Regresión:
- Linear Regression
- Ridge/Lasso
- Random Forest Regression
- Gradient Boosting Regression
- SVR

### Outputs Generados:
```
models/
├── logistic_regression.pkl
├── random_forest.pkl
├── gradient_boosting.pkl
└── ...

outputs/
├── evaluation_metrics.csv
├── feature_importance.png
├── confusion_matrix.png
├── roc_curve.png
└── predictions.csv
```

---

## 📊 Métricas de Evaluación

*Completar después de entrenar modelos*

### Clasificación:
| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | ? | ? | ? | ? | ? |
| Random Forest | ? | ? | ? | ? | ? |
| Gradient Boosting | ? | ? | ? | ? | ? |

### Regresión:
| Modelo | MAE | MSE | RMSE | R² |
|--------|-----|-----|------|-----|
| Linear Regression | ? | ? | ? | ? |
| Random Forest | ? | ? | ? | ? |
| Gradient Boosting | ? | ? | ? | ? |

---

## 🎨 Visualizaciones Disponibles

Accesibles en `ui/app.py`:
- Matriz de confusión
- Curva ROC
- Feature importance
- Predicciones vs Actuals
- Distribución de residuales

---

## ✅ Validación de Modelos

Ejecutar:
```bash
# Entrenar modelos
python notebooks/04_Model_Training.ipynb

# Visualizar resultados
streamlit run ui/app.py
```

---

## 🚀 Próximos Pasos

Modelos entrenados y evaluados.
Proceder con:
- Ajuste de hiperparámetros
- Cross-validation
- Selección del mejor modelo
- Preparación para producción
