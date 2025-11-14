# AGENT.md — Guía del Agente para el Proyecto ML (Educación Superior Chile)

Fecha: 2025-11-14T00:44:39.692Z

## Contexto y objetivo
Proyecto de Modelos Predictivos (Clasificación de Modalidad y Regresión de Edad Promedio) con arquitectura modular: `Config` centraliza rutas/targets y `MLPipeline` orquesta fases (load → EDA → preprocess → feature engineering → train → evaluate → interpret).

## Árbol y puntos de entrada
- Código fuente: `src/` (pipeline.py, eda.py, preprocessing/clean.py, features/engineer.py, models/model_architecture.py)
- Datos: `data/raw/data_input.csv`, `data/processed/`
- Artefactos: `outputs/eda/*`, `outputs/metrics`, `models/{trained,metadata}`, `reports/final_report.pdf`
- CLI: `python scripts/run_pipeline.py`, segmentación: `python scripts/execute_pipeline.py`, verificación: `python scripts/verify_pipeline.py`
- UI: `streamlit run ui/pipeline_executor.py`
- Notebooks: `jupyter notebook notebooks/full_pipeline_run.ipynb`

## Preparación y limpieza
- Preparar entorno: `source venv/bin/activate && pip install -r requirements.txt`
- Dataset: colocar CSV en `data/raw/data_input.csv` (sep="," por defecto).
- Limpiar artefactos: `bash clean.sh` (recrea estructura vacía).

## Flujo de ejecución (desde cero)
1) CLI: `python scripts/run_pipeline.py` (orquesta full pipeline). Alternativas: `scripts/execute_pipeline.py --phase 1|2|all`, `scripts/verify_pipeline.py`.
2) UI: `streamlit run ui/pipeline_executor.py` (ejecución interactiva por pasos).
3) Notebook: abrir `notebooks/full_pipeline_run.ipynb` y ejecutar celdas que invocan `MLPipeline`.

## Convenciones para el agente
- Cambios mínimos y quirúrgicos; mantener compatibilidad con `Config` y `MLPipeline`.
- No tocar `data/raw` salvo lectura; no incluir secretos en código.
- Idioma: Español en mensajes/CLI/UI.
- Validar artefactos en `outputs`, `models`, `reports` tras cada fase.

## Referencias rápidas
- Configuración: `config/config.py`
- Pipeline: `src/pipeline.py`
- EDA: `src/eda.py`
- Preprocesamiento: `src/preprocessing/clean.py`
- Features: `src/features/engineer.py`
- Modelos: `src/models/model_architecture.py`


# Objetivos y requerimientos del proyecto
- Objetivos del proyecto: `docs/objetivos_proyecto.md`
- Requerimientos del proyecto: `docs/requeriminetos_proyecto.md`
- Arquitectura del proyecto: `docs/arquitectura_proyecto.md`
- Fases del proyecto a seguir: `docs/fases_proyecto.md`
