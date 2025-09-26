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
   python ejemplos.py
   ```

## Estructura del proyecto

- `data/`: Carpeta para almacenar datos CSV o datasets.
- `src/`: Código fuente Python.
- `notebooks/`: Notebooks para prototipado y exploración interactiva.
- `outputs/`: Resultados generados como gráficos y reportes.
- `requirements.txt`: Librerías necesarias para el entorno Python.
- `README.md`: Descripción y guía del proyecto.