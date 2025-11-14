#!/usr/bin/env bash
# clean.sh - Limpia artefactos generados y recrea estructura vacía
set -euo pipefail

echo "[clean] Iniciando limpieza de artefactos..."

# Directorios base según Config
OUTPUTS_DIR="outputs"
EDA_DIR="$OUTPUTS_DIR/eda"
METRICS_DIR="$OUTPUTS_DIR/metrics"
MODELS_DIR="models"
TRAINED_DIR="$MODELS_DIR/trained"
METADATA_DIR="$MODELS_DIR/metadata"
REPORTS_DIR="reports"
PROCESSED_DIR="data/processed"

# No se toca data/raw

remove_dir() {
  local dir="$1"
  if [ -d "$dir" ]; then
    echo "[clean] Eliminando $dir"
    rm -rf "$dir"
  fi
}

remove_dir "$EDA_DIR"
remove_dir "$METRICS_DIR"
remove_dir "$TRAINED_DIR"
remove_dir "$METADATA_DIR"
remove_dir "$REPORTS_DIR"
remove_dir "$PROCESSED_DIR"

# Recrear estructura vacía
mkdir -p "$EDA_DIR" "$METRICS_DIR" "$TRAINED_DIR" "$METADATA_DIR" "$REPORTS_DIR" "$PROCESSED_DIR"

echo "[clean] Estructura recreada:"
for d in "$EDA_DIR" "$METRICS_DIR" "$TRAINED_DIR" "$METADATA_DIR" "$REPORTS_DIR" "$PROCESSED_DIR"; do
  echo "  - $d"
  touch "$d/.gitkeep"
done

echo "[clean] Limpieza completada."