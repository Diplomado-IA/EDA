# Guía de Análisis - Dataset Titulados 2007-2024

## Descripción del Dataset

Dataset que contiene información sobre titulaciones universitarias en Chile entre 2007 y 2024.

**Ubicación**: `data/TITULADO_2007-2024_web_19_05_2025_E.csv`

**Características**:
- **Filas**: 218,566 registros
- **Columnas**: 42 variables
- **Separador**: Punto y coma (`;`)
- **Encoding**: Latin-1

## Variables Principales

### Demográficas
- `AÑO`: Año de titulación
- `REGIÓN`: Región de la institución
- `PROVINCIA`: Provincia
- `COMUNA`: Comuna

### Institucionales
- `NOMBRE INSTITUCIÓN`: Nombre de la universidad/instituto
- `CLASIFICACIÓN INSTITUCIÓN NIVEL 1/2/3`: Clasificación jerárquica de la institución
- `CÓDIGO INSTITUCIÓN`: Identificador único
- `NOMBRE SEDE`: Sede específica

### Académicas
- `NOMBRE CARRERA`: Nombre de la carrera
- `ÁREA DEL CONOCIMIENTO`: Área general
- `CINE-F_97 ÁREA/SUBAREA`: Clasificación internacional (1997)
- `CINE-F_13 ÁREA/SUBAREA`: Clasificación internacional (2013)
- `NIVEL GLOBAL`: Nivel de estudios (Pregrado, Postgrado, etc.)
- `MODALIDAD`: Presencial, No Presencial
- `JORNADA`: Diurna, Vespertina, A Distancia
- `DURACIÓN ESTUDIO CARRERA`: Duración en semestres
- `DURACIÓN TOTAL DE LA CARRERA`: Duración total

### Titulaciones
- `TOTAL TITULACIONES`: Total de titulados
- `TITULACIONES MUJERES POR PROGRAMA`: Tituladas mujeres
- `TITULACIONES HOMBRES POR PROGRAMA`: Titulados hombres
- `TITULACIONES NB E INDEFINIDO POR PROGRAMA`: Titulados no binarios/indefinidos

### Edad
- `RANGO DE EDAD 15 A 19 AÑOS` hasta `RANGO DE EDAD 40 Y MÁS AÑOS`
- `PROMEDIO EDAD PROGRAMA`: Edad promedio por programa
- `PROMEDIO EDAD MUJER/HOMBRE/NB`: Promedios por género

## Comandos de Análisis

### 1. Análisis Rápido (sin gráficos)

```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --no-show --no-histos --no-box
```

### 2. Análisis Completo con Variable Objetivo

```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --objetivo "REGIÓN" \
  --no-show
```

### 3. Análisis Enfocado en Categorías Específicas

```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --cat-cols "NOMBRE INSTITUCIÓN" "ÁREA DEL CONOCIMIENTO" "NIVEL GLOBAL" \
  --max-cats 15 \
  --no-show
```

### 4. EDA Mínimo Todo-en-Uno

```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --objetivo "ÁREA DEL CONOCIMIENTO" \
  --run-minimo \
  --no-show
```

### 5. Script Automatizado

```bash
./analizar_titulados.sh
```

## Preguntas de Negocio Sugeridas

### Temporales
1. ¿Cómo ha evolucionado el número de titulados por año?
2. ¿Hay tendencias en las áreas de conocimiento más populares?

### Institucionales
3. ¿Cuáles son las instituciones con más titulados?
4. ¿Cómo se distribuyen las titulaciones por tipo de institución?

### Geográficas
5. ¿Qué regiones concentran más titulaciones?
6. ¿Hay diferencias entre regiones en áreas de conocimiento?

### Género
7. ¿Cómo es la distribución de género por área de conocimiento?
8. ¿Qué carreras tienen mayor paridad/disparidad de género?

### Edad
9. ¿Cuál es el perfil etario típico por tipo de carrera?
10. ¿Hay diferencias de edad entre modalidades presencial/online?

### Duración y Modalidad
11. ¿Cómo se relaciona la duración de carreras con el área de conocimiento?
12. ¿Cuál es la distribución entre modalidades presencial y a distancia?

## Resultados Generados

Después de ejecutar el análisis, encontrarás:

### En `outputs/resumen/`:
- `carga_info.txt`: Metadatos de la carga del dataset
- `resumen_columnas.csv`: Resumen de calidad de todas las columnas
- `resumen_columnas_ordenado.csv`: Resumen ordenado por % faltantes
- `top10_faltantes.csv`: Top 10 columnas con más valores faltantes
- `descriptivos_numericos.csv`: Estadísticas descriptivas de variables numéricas
- `topcats_*.csv`: Top categorías por cada variable categórica
- `decision_metricas.txt`: Recomendaciones de métricas (si hay objetivo)

### En `outputs/figures/`:
- `objetivo_barras.png`: Distribución de la variable objetivo
- `histogramas_numericas.png`: Histogramas de todas las variables numéricas
- `boxplots_numericas.png`: Boxplots de todas las variables numéricas

## Análisis con Python

También puedes usar las funciones directamente en Python:

```python
from src.cargar_csv import cargar_csv
from src.eda import eda_minimo

# Cargar datos
df, metadata = cargar_csv("data/TITULADO_2007-2024_web_19_05_2025_E.csv", sep=";")

# Análisis completo en un paso
resultados = eda_minimo(
    df, 
    objetivo="REGIÓN",
    max_cats=10,
    no_show=True
)
```

## Notas Importantes

1. **Encoding**: El archivo usa Latin-1 (no UTF-8), ya detectado automáticamente
2. **Separador**: Usa punto y coma (`;`), no coma
3. **Tamaño**: 218K filas - el análisis puede tomar 1-2 minutos
4. **Memoria**: Asegúrate de tener al menos 2GB RAM disponible
5. **Valores Faltantes**: Varias columnas tienen valores faltantes, revisa `top10_faltantes.csv`

## Próximos Pasos

1. Revisar archivos en `outputs/resumen/` para entender la calidad de datos
2. Identificar columnas con muchos faltantes para decidir si imputar o eliminar
3. Analizar distribuciones en `outputs/figures/`
4. Crear visualizaciones específicas según preguntas de negocio
5. Preparar datos para modelado (si aplica)
