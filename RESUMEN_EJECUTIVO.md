# 📊 Resumen Ejecutivo - Dataset Titulados 2007-2024

## ✅ Proyecto Adaptado Exitosamente

El proyecto EDA ha sido **completamente adaptado** para analizar el nuevo dataset de titulados universitarios en Chile.

---

## 📁 Dataset Cargado

**Archivo**: `data/TITULADO_2007-2024_web_19_05_2025_E.csv`

| Métrica | Valor |
|---------|-------|
| **Registros** | 218,566 filas |
| **Variables** | 42 columnas |
| **Tamaño** | 103 MB |
| **Encoding** | Latin-1 (detectado automáticamente ✅) |
| **Separador** | `;` (punto y coma) |

---

## Hallazgos Principales

### 1. Distribución por Nivel de Estudios
- **Pregrado**: 172,204 titulaciones (78.8%)
- **Postítulo**: 23,405 titulaciones (10.7%)
- **Posgrado**: 22,957 titulaciones (10.5%)

### 2. Modalidad de Estudio
- **Presencial**: 176,795 (81.0%)
- **Sin información**: 26,554 (12.2%)
- **No Presencial**: 8,455 (3.9%)
- **Semipresencial**: 6,762 (3.1%)

### 3. Calidad de Datos

#### Columnas con Muchos Valores Faltantes:
1. **TITULACIONES NB E INDEFINIDO** - 100% faltantes
2. **PROMEDIO EDAD NB** - 100% faltantes
3. **RANGO DE EDAD SIN INFORMACIÓN** - 99.73% faltantes
4. **RANGO DE EDAD 15 A 19 AÑOS** - 98.44% faltantes
5. **RANGO DE EDAD 40 Y MÁS AÑOS** - 58.19% faltantes

**Recomendación**: Considerar eliminar columnas con >95% faltantes para análisis específicos.

---

## Cómo Usar el Proyecto

### Opción 1: Script Rápido
```bash
./analizar_titulados.sh
```

### Opción 2: Comando Manual
```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --objetivo "REGIÓN" \
  --no-show
```

### Opción 3: Análisis Personalizado
```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --cat-cols "NOMBRE INSTITUCIÓN" "ÁREA DEL CONOCIMIENTO" \
  --max-cats 15 \
  --objetivo "NIVEL GLOBAL" \
  --no-show
```

---

## Archivos Generados

### En `outputs/resumen/`
✅ `carga_info.txt` - Metadatos del dataset  
✅ `resumen_columnas.csv` - Calidad de 42 columnas  
✅ `top10_faltantes.csv` - Top columnas con faltantes  
✅ `descriptivos_numericos.csv` - Estadísticas numéricas  
✅ `topcats_*.csv` - Distribuciones categóricas (5 archivos)

### En `outputs/figures/` (si se habilitan gráficos)
📊 `objetivo_barras.png` - Distribución de variable objetivo  
📊 `histogramas_numericas.png` - Histogramas de variables numéricas  
📊 `boxplots_numericas.png` - Boxplots para detectar outliers

---

## 📚 Documentación Adicional

| Archivo | Descripción |
|---------|-------------|
| **GUIA_TITULADOS.md** | Guía completa: variables, comandos, preguntas de negocio |
| **CAMBIOS_DATASET.md** | Resumen de cambios realizados en el proyecto |
| **README.md** | Documentación general del proyecto EDA |
| **ejemplos.py** | Ejemplos de uso de funciones (actualizado) |

---

## 🎯 Variables Clave para Análisis

### Demográficas
- `AÑO`, `REGIÓN`, `PROVINCIA`, `COMUNA`

### Institucionales
- `NOMBRE INSTITUCIÓN`, `CLASIFICACIÓN INSTITUCIÓN`

### Académicas
- `NOMBRE CARRERA`, `ÁREA DEL CONOCIMIENTO`
- `NIVEL GLOBAL`, `MODALIDAD`, `JORNADA`

### Titulaciones
- `TOTAL TITULACIONES`
- `TITULACIONES MUJERES/HOMBRES POR PROGRAMA`

### Edad
- Rangos etarios (15-19, 20-24, 25-29, etc.)
- Promedios de edad por género

---

## 💡 Preguntas de Negocio Sugeridas

1. **Temporal**: ¿Cómo evolucionan las titulaciones por año?
2. **Geográfica**: ¿Qué regiones tienen más titulaciones?
3. **Género**: ¿Qué áreas tienen mayor paridad de género?
4. **Edad**: ¿Cuál es el perfil etario por tipo de carrera?
5. **Modalidad**: ¿Cómo ha crecido la educación a distancia?
6. **Institucional**: ¿Qué universidades lideran en titulaciones?

---

## ⚙️ Archivos Modificados

✏️ **ejemplos.py** - Actualizada ruta del CSV  
✏️ **README.md** - Documentación actualizada  
➕ **analizar_titulados.sh** - Script de ejecución  
➕ **GUIA_TITULADOS.md** - Guía completa del dataset  
➕ **CAMBIOS_DATASET.md** - Log de cambios  
➕ **RESUMEN_EJECUTIVO.md** - Este archivo

---

## ✅ Estado del Proyecto

| Item | Estado |
|------|--------|
| Carga de CSV | ✅ Funcionando |
| Detección de encoding | ✅ Automática (Latin-1) |
| Análisis de calidad | ✅ Completo |
| Generación de reportes | ✅ 10 archivos |
| Documentación | ✅ Actualizada |
| Scripts automatizados | ✅ Creados |

---

## 🔄 Próximos Pasos Sugeridos

1. **Explorar reportes** en `outputs/resumen/`
2. **Identificar columnas a limpiar** (revisar faltantes)
3. **Ejecutar análisis con objetivo** específico (ej: REGIÓN)
4. **Generar visualizaciones** específicas según necesidad
5. **Crear análisis temporal** si interesa evolución 2007-2024

---

## 📞 Comandos Útiles

```bash
# Ver estructura de archivos generados
ls -lh outputs/resumen/

# Ver info de carga
cat outputs/resumen/carga_info.txt

# Ver top de faltantes
cat outputs/resumen/top10_faltantes.csv

# Ejecutar análisis completo
./analizar_titulados.sh
```

---

**Proyecto listo para análisis de datos de titulados 2007-2024** 🎓✨
