````markdown
# Proyecto ML reestructurado según arquitectura modular

## 1) Descarga del proyecto

- Requiere **Git** y **Python 3.10+**.
- Clonar el repo y entrar al directorio del proyecto:

```bash
git clone <URL_DEL_REPO>
cd EDA
````

---

## 2) Configuración básica

* Crear y activar entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

* Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## 3) Dataset y configuración

* Verifica que el CSV esté en:

  ```text
  data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv
  ```

* Configuración actual:

  * `config/config.py` usa separador `';'` y *encoding* `'latin1'`.
  * Ajusta estos parámetros si cambias el archivo de entrada.

* Objetivos del modelo:

  * `MODALIDAD_BIN`

    * Clase 1 = **Presencial**
    * Clase 0 = **No presencial / otras modalidades**
  * `PROMEDIO_EDAD_PROGRAMA`

---

## 4) Ejecutar la UI

* Lanzar la aplicación **Streamlit**:

```bash
streamlit run ui/app.py
```

* En la UI encontrarás:

  * **Fase 1:**
    Validar objetivos y configuración.

  * **Fase 2 – EDA (Análisis Exploratorio de Datos):**

    * Carga el dataset.
    * Ejecuta el EDA.
    * Visualiza artefactos (`.csv` / `.png`) indicando su ruta.

  * **Fase 3 – Preprocesamiento:**

    * Limpieza.
    * *Split* temporal.
    * Escalado (**StandardScaler**).
    * Codificación segura:

      * One-Hot Encoding (OHE) con *rare grouping* / *frequency encoding*.
    * *Features*:

      * HHI
      * LQ
      * IPG
    * Cálculo de **correlación** y **VIF** optimizados.
    * Selección de variables y guardado de resultados.

  * **Informes:**
    Pestañas con todos los `.md` de `docs/` renderizados.

  * Botón lateral:

    * **"Limpiar artefactos (clean.sh)"** para reiniciar la salida del proyecto sin tocar los datos crudos.

---

## 5) Artefactos generados

* **EDA / Resúmenes:**

  * `outputs/eda/resumen/*`
    (CSV, `decision_metricas.txt`)

* **Correlación / VIF:**

  * `data/processed/correlation_matrix.csv`
  * `data/processed/vif_scores.csv`
  * Archivo auxiliar: `*columns_used.txt`

* **Selección de features:**

  * `data/processed/selected_features.txt`

* **Datasets finales:**

  * `data/processed/X_train_engineered.csv`
  * `data/processed/X_test_engineered.csv`

* **Gráficos:**

  * `outputs/eda/figures/*`
  * Copias en: `data/processed/*.png`

---

## Notas

* **ML** = *Machine Learning* (Aprendizaje Automático).

* **OHE** = *One-Hot Encoding*.

* **VIF** = *Variance Inflation Factor*.

* Si cambias los objetivos (`MODALIDAD_BIN` / `PROMEDIO_EDAD_PROGRAMA`), actualiza:

  * `config/config.py`.

* `clean.sh`:
bash clean.sh
  * Recrea la estructura de artefactos vacía.
  * **No** modifica `data/raw`.

* Para limitar uso de CPU en cálculos intensivos:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Ejecución desde CLI

- Activar entorno e instalar dependencias:
  
  ```bash
  python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  ```

- Ejecutar flujo completo (EDA + preprocesamiento):
  
  ```bash
  python scripts/run_all.py
  ```

- Artefactos generados:
  - data/processed/* (datasets, correlación, VIF, features seleccionadas)
  - outputs/eda/resumen/* (resúmenes de preprocesamiento)
