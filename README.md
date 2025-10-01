# Descripción y guía del proyecto

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
   ```
   Parámetros CLI

   - `--csv` (obligatorio): ruta al CSV de entrada.

   - `--sep` (opcional): separador, por defecto `,`.

   - `--objetivo` (opcional): nombre de la columna objetivo para analizar distribución.

   - `--no-show` (flag): no abre la ventana de matplotlib; solo guarda el gráfico.

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
- `requirements.txt`: Librerías necesarias para el entorno Python.
- `README.md`: Descripción y guía del proyecto.