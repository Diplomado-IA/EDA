#!/usr/bin/env bash
set -euo pipefail
TS="2025-11-21T02:53:10.359Z"
echo "[INFO] Iniciando flujo completo ($TS)"

# 1. Limpiar artefactos
bash clean.sh

# 2. Pipeline principal con HPO
python3 scripts/run_all.py --with-hpo --hpo-method=grid

# 3. HPO adicional (clasificación y regresión rápidos)
python3 scripts/run_hpo.py --task clf --method grid --fast --out-dir outputs/hpo_clf_fast
python3 scripts/run_hpo.py --task reg --method grid --fast --out-dir outputs/hpo_reg_fast_opt

# 4. Clustering
python3 scripts/run_clustering.py --save-plots --pca --max-features 40 --seed 42 --sil-sample 3000 --sample-size 40000

# 5. Anomalías
echo "[INFO] Ejecutando detección de anomalías"
python3 scripts/run_anomaly_detection.py --save-plots --pca --max-features 40 --seed 42 --sample-size 40000 --contamination 0.05

# 6. Justificación de modelo
python3 scripts/generate_model_justification.py

# 7. Presentación (PDF; PPTX si dependencia existe)
python3 scripts/generate_presentation.py --export-pdf --export-pptx || echo "[WARN] PPTX pudo no generarse (falta dependencia)"

echo "[INFO] Flujo completo finalizado"
