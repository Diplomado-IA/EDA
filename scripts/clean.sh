#!/usr/bin/env bash
# Script para limpiar artefactos generados y reiniciar estructura
set -euo pipefail

echo "[CLEAN] Eliminando artefactos generados..."
rm -rf data/processed outputs/eda outputs/resumen outputs/figures models/trained models/metadata reports || true

# Eliminar pickles sueltos (si quedaran fuera de directorios)
find data -type f -name '*_engineered.pkl' -delete || true
find data -type f -name 'X_*_engineered.pkl' -delete || true
find data -type f -name 'y_*_classification.pkl' -delete || true
find data -type f -name 'y_*_regression.pkl' -delete || true

echo "[CLEAN] Recreando estructura vacía..."
mkdir -p data/processed outputs/eda outputs/resumen outputs/figures models/trained models/metadata reports

echo "[CLEAN] Limpieza completa. Proyecto listo para reiniciar el pipeline."
