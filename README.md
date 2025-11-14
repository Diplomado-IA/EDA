Proyecto ML reestructurado segun arquitectura modular (CLI, UI, Notebooks). Ver config/config.py y src/pipeline.py.

Guía rápida para descargar y ejecutar la UI (actualizado 2025-11-14T03:48:30.354Z)

1) Descarga del proyecto
- Requiere Git y Python 3.10+.
- Clonar el repo y entrar al directorio del proyecto:
  git clone <URL_DEL_REPO>
  cd EDA

2) Configuración básica
- Crear y activar entorno virtual:
  python3 -m venv venv
  source venv/bin/activate
- Instalar dependencias:
  pip install -r requirements.txt

3) Dataset y configuración
- Verifica que el CSV grande esté en: data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv
- Config actual: config/config.py usa separador ';' y encoding 'latin1'. Ajusta si cambias el archivo.

4) Ejecutar la UI
- Lanzar la aplicación Streamlit:
  streamlit run ui/app.py
- En la UI:
  - Fase 1: valida objetivos y configuración.
  - Fase 2: pulsa "Cargar Dataset" y luego "Ejecutar EDA" (genera artefactos y gráficos Fase 1).
  - Usa el botón lateral "Limpiar artefactos (clean.sh)" para reiniciar.

5) Artefactos generados
- Resúmenes/EDA: outputs/eda/resumen/* (CSV, decision_metricas.txt)
- Gráficos: outputs/eda/figures/* y copias en data/processed/*.png

6) CLI opcional
- Ejecutar Fase 1 por CLI:
  ./venv/bin/python scripts/execute_pipeline.py --phase 1
- Ejecutar verificación:
  ./venv/bin/python scripts/verify_pipeline.py

Notas
- Si cambias objetivos (MODALIDAD_BIN / PROMEDIO EDAD PROGRAMA ), actualiza config/config.py.
- clean.sh recrea la estructura vacía sin tocar data/raw.
