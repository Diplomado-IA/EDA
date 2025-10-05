# TODO — Actividad EDA Grupal

Guía de tareas para completar la EDA según `docs/02_Actividad_EDA_Grupal.pptx`, con el estado actual del código y los entregables esperados.

---

## 0) Estado global (hoy)

- [x] Carga robusta de CSV (validación ruta, encodings, `sep`) — `src/cargar_csv.py`
- [x] Info básica (`head`, `dtypes`, shape) — `src/eda.py::mostrar_info_basica`
- [x] Resumen de columnas (dtype, n_missing, %missing, n_unique, maybe_binary_text) — `src/resumen_columnas.py`
- [x] Distribución objetivo + gráfico + IR — `src/eda.py::plot_distrib_objetivo`
- [x] **Ordenar** resumen (%missing desc, n_unique asc) y exportar vistas top-10 faltantes
- [x] **Descriptivos numéricos** (`describe().T`) exportados
- [x] **Top categorías** por columna (3–5 columnas representativas) exportados
- [x] **Histogramas numéricos** guardados en `outputs/figures/`
- [x] **Boxplots** horizontales guardados en `outputs/figures/`
- [x] **Función EDA mínimo** que orqueste todo y deje outputs consolidados
- [x] **Decisión de métrica/estrategia** persistida en archivo
- [ ] **Informe breve** (2–3 págs / Markdown) con 3 hallazgos
- [ ] Revisión de **datos sensibles / anonimización** y **cita de fuente**

---

## 1) Carga y evidencia

- [x] Validar existencia de archivo (`Path.exists`)
- [x] Probar encodings (`utf-8`, `latin-1`; opcional: `cp1252`)
- [x] Aceptar separador por CLI (`--sep`)
- [x] Imprimir filas/columnas y `dtypes`
- [x] Persistir **evidencia de carga** (ruta, separador, encoding, `len(df)`, `shape`) en `outputs/resumen/carga_info.txt`

**Acción**

- [x] `src/main.py`: escribir `outputs/resumen/carga_info.txt` con metadatos de carga.

---

## 2) Calidad de datos (resumen de columnas)

- [x] Tabla: `dtype`, `n_missing`, `%missing`, `n_unique`, `maybe_binary_text`
- [x] Orden: por `%missing` desc, luego `n_unique` asc
- [x] Exportar `resumen_columnas_ordenado.csv`
- [x] Exportar `top10_faltantes.csv`
- [ ] (Opcional) Markdown con comentarios de imputación/descarte por columna

**Acciones**

- [x] `src/main.py`: ordenar y exportar vistas.
- [x] (Opcional) `src/resumen_columnas.py`: helper `ordenar_resumen(df)`.

---

## 3) Descriptivos y categóricas

- [x] `descriptivos_numericos.csv` (`describe().T`)
- [x] Top-10 categorías para 3–5 columnas categóricas (o las indicadas por CLI)
- [x] Exportar `topcats_<col>.csv` por cada columna seleccionada

**Acciones**

- [x] `src/eda.py`: `descriptivos_numericos(df) -> pd.DataFrame` + export en `main`.
- [x] `src/eda.py`: `top_categorias(df, columnas=None, k=10) -> dict[col, Series]` + export.

---

## 4) Variable objetivo y desbalance

- [x] Conteos y proporciones
- [x] IR (mayor/menor)
- [x] Gráfico de barras `outputs/figures/objetivo_barras.png`
- [x] Persistir **decisión de métrica/estrategia** según IR

**Acción**

- [x] `src/main.py`: crear `outputs/resumen/decision_metricas.txt`
  - IR ≥ 1.5 → _F1/AUC-PR, estratificación, class_weight/resampling_
  - Balanceado → _Accuracy/ROC-AUC_ base

---

## 5) Gráficos numéricos

- [x] Histogramas para todas las numéricas → `outputs/figures/hist_<col>.png`
- [x] Boxplots horizontales → `outputs/figures/box_<col>.png`
- [x] Flags CLI para omitir: `--no-histos`, `--no-box`

**Acciones**

- [x] `src/eda.py`: `graficar_histogramas(df, max_cols=12)`
- [x] `src/eda.py`: `graficar_boxplots(df, max_cols=12)`
- [x] `src/main.py`: integrar flags y llamadas.

---

## 6) EDA mínimo (todo en uno)

- [x] `eda_minimo(df, objetivo=None, max_cats=10, no_show=True)` que:
  - [x] Genere y **ordene** el resumen de columnas (+ top-10 faltantes)
  - [x] Exporte descriptivos numéricos
  - [x] Exporte top de categorías
  - [x] (Si `objetivo`) calcule IR, grafique barras y guarde decisión de métricas
  - [x] (Opcional) Genere histos/boxplots según flags

**Acciones**

- [x] `src/eda.py`: implementar `eda_minimo(...)`
- [x] `src/main.py`: flag `--run-minimo` que lo invoque.

---

## 7) Entregables

- [x] Carpeta `outputs/` con:
  - [x] `resumen/resumen_columnas.csv`
  - [x] `resumen/resumen_columnas_ordenado.csv`
  - [x] `resumen/top10_faltantes.csv`
  - [x] `resumen/descriptivos_numericos.csv`
  - [x] `resumen/topcats_*.csv`
  - [x] `resumen/decision_metricas.txt`
  - [x] `figures/objetivo_barras.png`
  - [x] `figures/hist_*.png`
  - [x] `figures/box_*.png`
- [ ] Informe breve (Markdown `outputs/INFORME_EDA.md` o PDF exportado)
  - [ ] Descripción del CSV y fuente
  - [ ] Tabla de calidad
  - [ ] Descriptivos/tops
  - [ ] Distribución objetivo + IR + decisión de métrica
  - [ ] **3 hallazgos** principales

**Acciones**

- [ ] `outputs/INFORME_EDA.md` (plantilla básica) — manual o generado desde `main` (opcional).
      Nota: Existe un archivo `outputs/resumen/generative_ai_tools_report.md` que está vacío y podría utilizarse para este propósito.

---

## 8) Reglas y compliance

- [ ] Verificar que no haya datos sensibles / anonimizar si aplica
- [ ] Citar fuente si el dataset es público
- [ ] Documentar en `INFORME_EDA.md` una línea de compliance

---

## 9) CLI / Usabilidad

- [x] `python -m src.main --csv datos.csv --sep ","`
- [x] `--objetivo <col>` y `--no-show`
- [x] `--max-cats`, `--cat-cols`
- [x] `--no-histos`, `--no-box`, `--run-minimo`
- [x] `outputs/resumen/carga_info.txt`

**Acciones**

- [x] `src/main.py`: agregar flags y llamadas nuevas.

---

## 10) Validación (check final)

Ejecutar desde la raíz del proyecto:

```powershell
# EDA tabular básica
python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep ","

# Con objetivo (sin mostrar gráfico en pantalla)
python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep "," --objetivo company --no-show

# EDA mínimo todo-en-uno (cuando esté implementado)
python -m src.main --csv "Generative AI Tools - Platforms 2025.csv" --sep "," --objetivo company --no-show --run-minimo --max-cats 10
```
