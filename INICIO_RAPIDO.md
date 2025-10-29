# 🚀 Guía de Inicio Rápido

## Para Empezar en 5 Minutos

### 1. Lee la Documentación Principal

```bash
# Empieza aquí
cat README_PROYECTO.md

# Luego revisa
cat RESUMEN_FINAL.txt
```

### 2. Activa el Entorno

```bash
source venv/bin/activate
```

### 3. Ejecuta el Pipeline Completo

```python
# Ejecuta esto en Python o en un notebook
from src.data.loader import load_titulados_data
from src.data.splitter import split_titulados_data
from src.data.preprocessor import preprocess_titulados_data
from src.features.engineer import create_titulados_features

# Cargar datos
print("📊 Cargando datos...")
df, metadata = load_titulados_data()

# Particionar
print("✂️ Particionando temporalmente...")
train_df, val_df, test_df = split_titulados_data(df)

# Preprocesar
print("🔧 Preprocesando SIN data leakage...")
train_processed, preprocessor = preprocess_titulados_data(train_df, fit=True)
val_processed, _ = preprocess_titulados_data(val_df, fit=False, preprocessor=preprocessor)
test_processed, _ = preprocess_titulados_data(test_df, fit=False, preprocessor=preprocessor)

# Crear features
print("✨ Creando features...")
train_features, engineer = create_titulados_features(train_processed)
val_features, _ = create_titulados_features(val_processed)
test_features, _ = create_titulados_features(test_processed)

print("\n✅ Pipeline completo ejecutado correctamente!")
print(f"Train: {len(train_features):,} filas")
print(f"Val:   {len(val_features):,} filas")
print(f"Test:  {len(test_features):,} filas")
```

---

## 📚 Archivos para Leer (en orden)

1. **RESUMEN_FINAL.txt** ← Empieza aquí (vista rápida)
2. **README_PROYECTO.md** ← Guía completa del proyecto
3. **IMPLEMENTACION_COMPLETA.md** ← Detalles técnicos
4. **docs/PROJECT_STRUCTURE.md** ← Entender las 10 fases
5. **docs/DATA_DICTIONARY.md** ← Conocer las variables

---

## 🎯 Qué se ha Implementado

✅ **Fase 0**: Exploración inicial  
✅ **Fase 1**: Particiones temporales sin data leakage  
✅ **Fase 2**: Feature engineering completo  
⏳ **Fases 3-10**: Por implementar

---

## 🏃 Próximo Paso

Continuar con **Fase 3: Métricas de Evaluación**

---

**¿Dudas?** Lee `README_PROYECTO.md` para entender el proyecto completo.
