# Pipeline EDA - Análisis Exploratorio de Datos

Este proyecto implementa un pipeline de análisis exploratorio de datos (EDA) para conjuntos de datos tabulares. Incluye funcionalidades para carga robusta de CSV, resumen de calidad de columnas, estadísticas descriptivas, visualización de distribuciones y análisis de variables objetivo.

## Cómo arrancar el proyecto

1. **Clonar el repositorio**:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```

2. **Navegar al directorio del proyecto**:

   ```bash
   cd pipeline
   ```

3. **Crear y activar un entorno virtual (opcional pero recomendado)**:

   ```bash
   python -m venv venv
   # Activar el entorno virtual:
   # En Windows
   .\venv\Scripts\activate
   # En macOS/Linux
   source venv/bin/activate
   ```

4. **Instalar las dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Ejecutar el script principal**:

   ```bash
   # Ejemplo completo:
   python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep "," --objetivo company

   # Sin mostrar el gráfico (solo guardar a archivo):
   python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep "," --objetivo company --no-show

   # Solo EDA tabular, sin objetivo:
   python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep ","

   # Especificando columnas categóricas y máximo de categorías:
   python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep "," --cat-cols category_canonical modality_canonical --max-cats 5

   # Sin generar histogramas ni boxplots:
   python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep "," --no-histos --no-box

   # EDA mínimo todo-en-uno:
   python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep "," --objetivo company --no-show --run-minimo
   ```

   Parámetros CLI

   - `--csv` (obligatorio): Ruta al CSV de entrada.
   - `--sep` (opcional): Separador, por defecto `,`.
   - `--objetivo` (opcional): Nombre de la columna objetivo para analizar distribución y calcular imbalance ratio.
   - `--no-show` (flag): No abre la ventana de matplotlib; solo guarda el gráfico.
   - `--cat-cols` (opcional): Columnas categóricas específicas para mostrar el top de categorías (separadas por espacios).
   - `--max-cats` (opcional): Número máximo de categorías a mostrar por columna categórica (por defecto 10).
   - `--no-histos` (flag): No generar histogramas para las variables numéricas.
   - `--no-box` (flag): No generar boxplots para las variables numéricas.
   - `--run-minimo` (flag): Ejecutar el EDA mínimo completo (todas las funciones en un solo paso).

## Estructura del proyecto

```bash
.
├── Generative AI Tools - Platforms 2025.csv   # Tu dataset
├── outputs/
│   ├── figures/                   # gráficos generados
│   └── resumen/                   # reportes/tablas
├── src/
│   ├── __init__.py                # convierte a paquete Python
│   ├── cargar_csv.py              # capa de datos
│   ├── resumen_columnas.py        # capa de calidad/EDA tabular
│   ├── eda.py                     # capa de EDA visual
│   └── main.py                    # orquestación/CLI
├── ejemplos.py                    # referencia (no se ejecuta)
├── requirements.txt
└── README.md

```

- `data/`: Carpeta para almacenar datos CSV o datasets.
- `src/`: Código fuente Python.
- `notebooks/`: Notebooks para prototipado y exploración interactiva.
- `outputs/`: Resultados generados como gráficos y reportes.
  - `figures/`: Visualizaciones (histogramas, boxplots, gráficos de barras)
  - `resumen/`: Archivos CSV y TXT con análisis tabulares y descriptivos
- `requirements.txt`: Librerías necesarias para el entorno Python.
- `TODO.md`: Lista de tareas pendientes y completadas.
- `README.md`: Descripción y guía del proyecto.

## Flujo de trabajo del EDA

El pipeline de análisis exploratorio sigue este flujo:

1. **Carga robusta del CSV**: Validación de ruta, detección automática de encoding y manejo de separador.
2. **Resumen de calidad de columnas**: Análisis de tipos de datos, valores faltantes, valores únicos y posibles columnas binarias.
3. **Descriptivos numéricos**: Estadísticas descriptivas completas (media, desviación, percentiles, asimetría, curtosis).
4. **Análisis de top categorías**: Identificación de las categorías más frecuentes en variables categóricas.
5. **Análisis de variable objetivo**: Evaluación de distribución, imbalance ratio y recomendación de métricas/estrategias.
6. **Visualización**: Generación de histogramas y boxplots para variables numéricas.

Todos los resultados se guardan en la carpeta `outputs/` para su posterior análisis.

## Función EDA mínimo

El proyecto incluye una función integral `eda_minimo()` que ejecuta todas las etapas del análisis exploratorio en un solo paso:

```python
# Desde Python
from src.eda import eda_minimo
resultados = eda_minimo(df, objetivo="columna_objetivo", max_cats=5)
```

O desde la línea de comandos:

```bash
python -m src.main --csv "mi_dataset.csv" --sep "," --objetivo "columna_objetivo" --run-minimo
```

Esta función es ideal para obtener rápidamente una visión completa de un dataset nuevo.
