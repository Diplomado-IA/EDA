
# 🚀 Proyecto ML – Arquitectura Modular con UI en Streamlit

Este repositorio contiene un flujo completo de **EDA → Preprocesamiento → Artefactos ML**, expuesto a través de una **UI interactiva en Streamlit** y estructurado según una **arquitectura modular**.


## 📦 1) Descarga del proyecto

### Requisitos previos

- **Git**
- **Python 3.10+**

### Clonar el repositorio

```bash
git clone <URL_DEL_REPO>
cd EDA
```

> 💡 Asegúrate de estar en la carpeta raíz del proyecto antes de continuar.


## 🛠️ 2) Configuración básica

### Crear y activar entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
# En Windows:
# venv\Scripts\activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```


## 📂 3) Dataset y configuración

### Ubicación del dataset

Verifica que el archivo CSV esté en:

```text
data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv
```

### Configuración actual

El archivo de configuración principal es:

```text
config/config.py
```

Allí se definen, entre otros:

* Separador del CSV: `';'`
* *Encoding*: `'latin1'`

> ⚙️ Si cambias el archivo de entrada o su formato, **ajusta estos parámetros** en `config/config.py`.

### Objetivos del modelo

* **Clasificación (`MODALIDAD_BIN`)**

  * Clase `1` → **Presencial**
  * Clase `0` → **No presencial / otras modalidades**

* **Regresión (`PROMEDIO_EDAD_PROGRAMA`)**

  * Variable continua de edad promedio por programa.



## 🎛️ 4) Ejecutar la UI (Streamlit)

### Lanzar la aplicación

```bash
streamlit run ui/app.py
```

### Secciones disponibles en la UI

* **Fase 1 – Configuración inicial**

  * Validar objetivos (`MODALIDAD_BIN`, `PROMEDIO_EDAD_PROGRAMA`).
  * Verificar ruta y parámetros de lectura del dataset.

* **Fase 2 – EDA (Análisis Exploratorio de Datos)**

  * Carga del dataset.
  * Ejecución del EDA automatizado.
  * Visualización de artefactos generados (`.csv`, `.png`) con su ruta correspondiente.

* **Fase 3 – Preprocesamiento**

  * Limpieza de datos.
  * *Split* temporal.
  * Escalado con **StandardScaler**.
  * Codificación segura de variables categóricas:

    * **One-Hot Encoding (OHE)** con *rare grouping* / *frequency encoding*.
  * Generación y cálculo de *features*:

    * **HHI**
    * **LQ**
    * **IPG**
  * Cálculo optimizado de:

    * **Matriz de correlación**
    * **VIF (Variance Inflation Factor)**
  * Selección de variables y guardado de resultados.

* **Fase 4 – Interpretabilidad (XAI)**

  * Entrena un modelo demo (RandomForest/Logistic/Linear) sobre train.
  * Explicabilidad: Feature Importance (árbol), Permutation Importance y Coeficientes lineales.
  * Guarda artefactos en `reports/*.csv` y muestra tablas/gráficos en la UI.

* **Informes**

  * Pestañas que renderizan todos los `.md` dentro de `docs/`.

* **Botón lateral**

  * **"Limpiar artefactos (clean.sh)"**
    Permite reiniciar la salida del proyecto sin modificar los datos crudos en `data/raw`.


## 📁 5) Artefactos generados

### EDA / Resúmenes

* `outputs/eda/resumen/*`
  Incluye:

  * CSVs de resumen
  * `decision_metricas.txt`

### Gráficos

* `outputs/eda/figures/*`
* Copias auxiliares en:

  * `data/processed/*.png`

### Correlación / VIF

* `data/processed/correlation_matrix.csv`
* `data/processed/vif_scores.csv`
* Archivos auxiliares:

  * `*columns_used.txt` (columnas empleadas para los cálculos)

### Selección de *features*

* `data/processed/selected_features.txt`

### Datasets finales

* `data/processed/X_train_engineered.csv`
* `data/processed/X_test_engineered.csv`

### Interpretabilidad (XAI)

* `reports/feature_importance_*.csv`
* `reports/permutation_importance_*.csv`
* `reports/coefficients_linear_*.csv`




## 🧪 6) Ejecución desde CLI (flujo completo)

Si prefieres correr el flujo sin UI:

### 6.1 Activar entorno e instalar dependencias

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Ejecutar flujo completo (EDA + preprocesamiento)

```bash
python scripts/run_all.py
```

### 6.3 Artefactos generados vía CLI

* `data/processed/*`

  * Datasets procesados
  * Correlación
  * VIF
  * *Features* seleccionadas

* `outputs/eda/resumen/*`

  * Resúmenes de EDA y preprocesamiento


## 🧾 7) Notas y convenciones

* **ML** → *Machine Learning* (Aprendizaje Automático)
* **OHE** → *One-Hot Encoding*
* **VIF** → *Variance Inflation Factor*

Si cambias los objetivos (`MODALIDAD_BIN` / `PROMEDIO_EDAD_PROGRAMA`), recuerda actualizar:

* `config/config.py`

### Script de limpieza: `clean.sh`

```bash
bash clean.sh
```

* Recrea la estructura de artefactos **vacía**.
* **No modifica** el contenido de `data/raw`.

### Limitar uso de CPU en cálculos intensivos (opcional)

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```


## ✅ Resumen rápido

* Clona el repo y crea un entorno virtual.
* Ajusta `config/config.py` si cambias el dataset.
* Ejecuta la UI con `streamlit run ui/app.py` **o** usa `python scripts/run_all.py` desde CLI.
* Usa `clean.sh` para resetear artefactos sin tocar los datos crudos.



