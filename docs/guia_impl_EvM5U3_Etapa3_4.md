# Guía de implementación EvM5U3 – Etapa 3/4

Objetivo: implementar detección de fuga del target, validación temporal y HPO reproducible.

## Pasos
1) Configurar config/params.yaml (leakage, cv, split, hpo).
2) Usar src/features/leakage.py para detectar fuga y registrar reports/leakage_report.json.
3) Generar splits con src/data/splits.py (TimeSeriesSplit o hold-out cronológico).
4) Ejecutar GridSearch con src/models/train.py y persistir outputs/hpo/*.
5) Documentar resultados y decisiones en reports/ y actualizar docs/.

## Ejemplo (pseudocódigo)
```python
from src.features.leakage import detect_leakage, save_leakage_report
from src.data.splits import get_cv
from src.models.train import run_grid_search

# X, y preparados; cfg cargada desde params.yaml
rep = detect_leakage(df, target="PROMEDIO_EDAD_PROGRAMA", suspect_features=["PROMEDIO_EDAD_HOMBRE","PROMEDIO_EDAD_MUJER"], r2_threshold=0.9)
save_leakage_report(rep)
cv = get_cv({"kind":"time", "n_splits":5})
# estimator y param_grid definidos según cfg
run_grid_search(estimator, param_grid, X, y, cv, scoring=["neg_mean_absolute_error","neg_root_mean_squared_error","r2"]) 
```
