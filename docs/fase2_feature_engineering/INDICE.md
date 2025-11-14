# ⚙️ FASE 2.2: FEATURE ENGINEERING

## Índice de Documentación

### Archivo de Referencia: `FASE_2_FEATURE_ENGINEERING.md`
Documentación completa del feature engineering.

### Componentes:
- **Módulo**: `src/features/engineer.py`
- **Notebook**: `notebooks/03_Feature_Engineering.ipynb`
- **Pipeline**: Integración en `MLPipeline`

### Objetivos:
1. ✅ Análisis de correlaciones
2. ✅ Detección de multicolinealidad (VIF)
3. ✅ Selección de features relevantes
4. ✅ Eliminación de varianza baja
5. ✅ Generación de dataset optimizado

### Procesos:

#### 1. **Análisis de Correlación**
```python
corr_matrix = engineer.calculate_correlation_matrix(X)
# Detecta features con r > 0.8 (redundantes)
```

#### 2. **VIF (Variance Inflation Factor)**
```python
vif_scores = engineer.calculate_vif(X)
# Detecta multicolinealidad entre features
# Threshold: VIF > 10
```

#### 3. **Selección Univariada**
```python
selected = engineer.select_features_univariate(X, y, k=15)
# Ranking por F-score
# Top 15 features más relevantes
```

#### 4. **Filtrado de Varianza**
```python
X_filtered = engineer.remove_low_variance_features(X)
# Elimina features casi constantes
# Threshold: varianza < 0.01
```

### Outputs Generados:
```
data/processed/
├── correlation_matrix.png
├── vif_scores.csv
├── selected_features.txt
├── X_train_engineered.pkl
├── X_test_engineered.pkl
└── feature_engineering_report.txt
```

---

## 📊 Resumen de Optimización

*Completar después de ejecutar*

### Feature Selection:
| Etapa | Features Entrada | Features Salida | Removidas |
|-------|------------------|-----------------|-----------|
| Correlación | 40 | ? | ? |
| VIF | ? | ? | ? |
| Univariada | ? | 15 | ? |
| Varianza | 15 | ? | ? |

### Top 15 Features Seleccionados:
```
1. [feature_name] - F-score: X.XX
2. [feature_name] - F-score: X.XX
...
```

---

## ✅ Validación de Feature Engineering

Ejecutar:
```bash
python -c "
import pickle
with open('data/processed/X_train_engineered.pkl', 'rb') as f:
    X = pickle.load(f)
print(f'✓ Dataset engineered: {X.shape}')
print(f'✓ Features optimizados: {X.shape[1]}')
print(f'✓ Filas: {X.shape[0]}')
"
```

---

## 🚀 Próximos Pasos

Una vez completada esta fase:
→ Ir a `docs/fase3_modelos/` para entrenar modelos
