# Verificación Fase 2 (EDA)

## Objetivo
Generar artefactos EDA (resúmenes + gráficos + análisis objetivo) y justificar estrategia de métricas según desbalance.

## Pasos de Ejecución
```bash
# Activar entorno
source venv/bin/activate

# Ejecutar EDA mínimo (suponiendo MODALIDAD_BIN existe)
python - <<'PY'
from config.config import Config
from src.eda import cargar_csv, eda_minimo
import os
os.makedirs('outputs/eda/resumen', exist_ok=True)
os.makedirs('outputs/eda/figures', exist_ok=True)
df = cargar_csv(Config.DATASET_PATH)
eda_minimo(df, objetivo=Config.TARGET_CLASSIFICATION, no_show=True)
PY
```

## Artefactos esperados
- outputs/resumen/resumen_columnas.csv
- outputs/resumen/resumen_columnas_ordenado.csv
- outputs/resumen/top10_faltantes.csv
- outputs/resumen/descriptivos_numericos.csv (si hay numéricas)
- outputs/resumen/decision_metricas.txt
- outputs/figures/objetivo_barras.png
- Histogramas: outputs/figures/hist_*.png (≥1)
- Boxplots: outputs/figures/box_*.png (≥1)

## Comprobación
```bash
ls outputs/resumen | grep -E 'resumen_columnas|top10_faltantes|decision_metricas' && echo OK_RESUMEN
ls outputs/figures | grep objetivo_barras && echo OK_OBJETIVO
ls outputs/figures | grep hist_ | wc -l | awk '{if($1>0)print "OK_HIST"; else print "NO_HIST"}'
ls outputs/figures | grep box_ | wc -l | awk '{if($1>0)print "OK_BOX"; else print "NO_BOX"}'
cat outputs/resumen/decision_metricas.txt | head -n 5
```

## Criterios de Aceptación
- Todos los CSV de resumen generados.
- Gráfico de la variable objetivo creado.
- Archivo decision_metricas.txt contiene referencia a F1/AUC-PR si IR ≥ 1.5.

## Próximos Pasos
Implementar funciones de preprocesamiento en Fase 3.
