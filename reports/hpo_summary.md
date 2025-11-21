# Resumen HPO
Refit metric: roc_auc

## Top 5 por roc_auc
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_roc_auc|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|0.990731479565326|

## Top 5 por f1_macro
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_f1_macro|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|0.9508702644829804|

## Top 5 por accuracy
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_accuracy|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|0.9595744680851064|

## Mejor configuración
```json
{
  "best_params_": {
    "max_depth": 10,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100
  },
  "best_score_": 0.9907314795653261,
  "refit_metric": "roc_auc"
}
```
---

# Resumen HPO
Refit metric: roc_auc

## Top 5 por roc_auc
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_roc_auc|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|0.990731479565326|

## Top 5 por f1_macro
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_f1_macro|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|0.9508702644829804|

## Top 5 por accuracy
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_accuracy|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|0.9595744680851064|

## Mejor configuración
```json
{
  "best_params_": {
    "max_depth": 10,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100
  },
  "best_score_": 0.9907314795653261,
  "refit_metric": "roc_auc"
}
```
---

# Resumen HPO
Refit metric: neg_root_mean_squared_error

## Top 5 por neg_mean_absolute_error
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_neg_mean_absolute_error|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|-0.5683471867719216|

## Top 5 por neg_root_mean_squared_error
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_neg_root_mean_squared_error|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|-0.7831218356244183|

## Top 5 por r2
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_r2|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|0.4888208295626315|

## Mejor configuración
```json
{
  "best_params_": {
    "max_depth": 10,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100
  },
  "best_score_": -0.5683471867719216,
  "refit_metric": "neg_mean_absolute_error"
}
```