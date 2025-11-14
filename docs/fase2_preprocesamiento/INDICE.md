# 🧹 FASE 2.1: PREPROCESAMIENTO DE DATOS

## Índice de Documentación

### Archivo de Referencia: `FASE_2_PREPROCESAMIENTO.md`
Documentación completa del proceso de preprocesamiento.

### Componentes:
- **Módulo**: `src/preprocessing/preprocessor.py`
- **Notebook**: `notebooks/02_Preprocesamiento.ipynb`
- **Pipeline**: Integración en `MLPipeline`

### Objetivos:
1. ✅ Manejar valores nulos
2. ✅ Detectar y tratar outliers
3. ✅ Normalizar/Escalar datos
4. ✅ Codificar variables categóricas
5. ✅ Generar dataset preprocesado

### Técnicas Aplicadas:
- Imputación de valores faltantes
- Detección de outliers (IQR, Z-score)
- Normalización y estandarización
- Encoding de variables categóricas
- Filtrado de varianza baja

### Outputs Generados:
```
data/processed/
├── X_train_preprocessed.pkl
├── X_test_preprocessed.pkl
├── y_train.pkl
├── y_test.pkl
├── preprocessing_log.txt
└── estadisticas_preprocesamiento.csv
```

---

## 📊 Estadísticas de Preprocesamiento

*Completar después de ejecutar*

### Antes vs Después:
| Métrica | Antes | Después |
|---------|-------|---------|
| Filas | 173,522 | ? |
| Columnas | 40 | ? |
| Valores nulos | ? | ? |
| Outliers removidos | ? | ? |
| Varianza mínima | ? | ? |

---

## ✅ Validación de Preprocesamiento

Ejecutar:
```bash
python -c "
import pickle
with open('data/processed/X_train_preprocessed.pkl', 'rb') as f:
    X = pickle.load(f)
print(f'✓ Dataset preprocesado: {X.shape}')
print(f'✓ Tipo: {type(X)}')
print(f'✓ Sin nulos: {X.isnull().sum().sum() == 0}')
"
```

---

## 🚀 Próximos Pasos

Una vez completada esta fase:
→ Ir a `docs/fase2_feature_engineering/` para optimizar features
