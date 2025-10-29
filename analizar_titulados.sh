#!/bin/bash
# Script para ejecutar análisis del dataset de titulados 2007-2024

echo "=== Análisis EDA - Titulados 2007-2024 ==="
echo ""

# Activar entorno virtual
source venv/bin/activate

# Análisis completo con variable objetivo REGIÓN
echo "Ejecutando análisis completo..."
python -m src.main \
  --csv "data/TITULADO_2007-2024_web_19_05_2025_E.csv" \
  --sep ";" \
  --objetivo "REGIÓN" \
  --cat-cols "NOMBRE INSTITUCIÓN" "ÁREA DEL CONOCIMIENTO" "NIVEL GLOBAL" "MODALIDAD" \
  --max-cats 10 \
  --no-show

echo ""
echo "=== Análisis completado ==="
echo "Revisa los resultados en:"
echo "  - outputs/resumen/    (reportes CSV y TXT)"
echo "  - outputs/figures/    (gráficos PNG)"
