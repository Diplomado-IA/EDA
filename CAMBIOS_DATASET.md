# Resumen de Cambios - Adaptación para Dataset Titulados

## ✅ Cambios Realizados

### 1. **ejemplos.py** - Actualizada ruta del CSV
- Cambió de: `cargar_csv("Generative AI Tools - Platforms 2025.csv")`
- A: `cargar_csv("data/TITULADO_2007-2024_web_19_05_2025_E.csv", sep=";")`
- Incluye separador de punto y coma (`;`)

### 2. **README.md** - Actualizada documentación
- Ejemplos de comandos actualizados con la nueva ruta
- Referencias al dataset de titulados 2007-2024
- Estructura de directorios actualizada

### 3. **Nuevos Archivos Creados**

#### `analizar_titulados.sh`
Script bash para ejecutar el análisis completo del dataset con un solo comando:
```bash
./analizar_titulados.sh
```

#### `GUIA_TITULADOS.md`
Guía completa que incluye:
- Descripción detallada del dataset (218K filas, 42 columnas)
- Documentación de todas las variables
- Comandos de análisis específicos
- Preguntas de negocio sugeridas
- Notas sobre encoding (Latin-1) y separador (`;`)

### 4. **Pruebas Realizadas**
✅ Carga exitosa del CSV con encoding Latin-1
✅ Generación de reportes en `outputs/resumen/`
✅ 10 archivos generados correctamente

## 📊 Dataset Actual

**Archivo**: `data/TITULADO_2007-2024_web_19_05_2025_E.csv`
- **Tamaño**: 103 MB
- **Filas**: 218,566 registros
- **Columnas**: 42 variables
- **Separador**: `;` (punto y coma)
- **Encoding**: Latin-1

## 🚀 Cómo Ejecutar

### Opción 1: Script automatizado (recomendado)
```bash
./analizar_titulados.sh
```

### Opción 2: Comando directo
```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --objetivo "REGIÓN" \
  --no-show
```

### Opción 3: Solo análisis tabular (rápido)
```bash
source venv/bin/activate
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --no-show --no-histos --no-box
```

## 📁 Resultados Generados

Los análisis se guardan en:
- `outputs/resumen/` - Reportes CSV y TXT
- `outputs/figures/` - Gráficos PNG

## 📖 Documentación

Consulta `GUIA_TITULADOS.md` para:
- Descripción completa de variables
- Preguntas de negocio sugeridas
- Ejemplos de análisis específicos
- Uso desde Python

## ⚠️ Notas Importantes

1. El proyecto detecta automáticamente el encoding Latin-1
2. Siempre usa `--sep ";"` para este dataset
3. El análisis completo toma 1-2 minutos (218K filas)
4. Los archivos anteriores del viejo dataset no se eliminaron
