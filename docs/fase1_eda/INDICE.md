# 📊 FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)

## Índice de Documentación

### Contenido de esta fase:
- **Notebook**: `notebooks/01_EDA.ipynb`
- **Generaciones**: Gráficos en `data/processed/`

### Objetivos:
1. ✅ Cargar y explorar dataset
2. ✅ Análisis univariado de variables
3. ✅ Detectar anomalías y patrones
4. ✅ Generar visualizaciones
5. ✅ Documentar hallazgos

### Variables Analizadas:
- PROMEDIO EDAD PROGRAMA
- GÉNERO PERSONERIA
- ESTADO CIVIL
- Otras variables demográficas

### Outputs Generados:
```
data/processed/
├── 01_carga_datos_estadisticas.png
├── 02_edad_distribucion.png
├── 03_genero_personeria.png
├── 04_correlation_matrix.png
└── ...
```

---

## 🔍 Hallazgos Clave

*Completar después de ejecutar EDA*

### Variables Target:
- [ ] PROMEDIO EDAD PROGRAMA
- [ ] GÉNERO PERSONERIA
- [ ] Otras métricas

### Anomalías Detectadas:
- [ ] Valores nulos
- [ ] Outliers
- [ ] Inconsistencias

---

## 📈 Gráficos Generados

Los gráficos están disponibles en `data/processed/`:
- Distribuciones
- Box plots
- Correlaciones
- Análisis de componentes

---

## ✅ Validación de Fase 1

Ejecutar:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv')
print(f'✓ Dataset cargado: {df.shape}')
print(f'✓ Gráficos en: data/processed/')
"
```

---

## 🚀 Próximos Pasos

Una vez completada esta fase:
→ Ir a `docs/fase2_preprocesamiento/` para limpiar datos
