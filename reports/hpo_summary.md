# Resumen HPO
Refit metric: roc_auc

## Top 5 por roc_auc
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_roc_auc|
|---|---|---|---|---|
|nan|2.0|5.0|100.0|0.9917664291060544|
|10.0|2.0|5.0|100.0|0.9917664291060544|
|10.0|2.0|2.0|200.0|0.9916778236281828|
|nan|2.0|2.0|200.0|0.9916778236281828|
|nan|2.0|5.0|200.0|0.9914403761900116|

## Top 5 por f1_macro
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_f1_macro|
|---|---|---|---|---|
|nan|1.0|2.0|100.0|0.9508702644829804|
|10.0|1.0|2.0|100.0|0.9508702644829804|
|10.0|1.0|5.0|200.0|0.9444698282761136|
|nan|1.0|5.0|200.0|0.9444698282761136|
|nan|1.0|2.0|200.0|0.9426226579783868|

## Top 5 por accuracy
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_accuracy|
|---|---|---|---|---|
|nan|1.0|2.0|100.0|0.9595744680851064|
|10.0|1.0|2.0|100.0|0.9595744680851064|
|nan|1.0|5.0|200.0|0.953191489361702|
|nan|1.0|2.0|200.0|0.953191489361702|
|10.0|1.0|2.0|200.0|0.953191489361702|

## Mejor configuración
```json
{
  "best_params_": {
    "max_depth": null,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "n_estimators": 100
  },
  "best_score_": 0.9917664291060543,
  "refit_metric": "roc_auc"
}
```
---

# Resumen HPO
Refit metric: roc_auc

## Top 5 por roc_auc
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_roc_auc|
|---|---|---|---|---|
|nan|2.0|5.0|100.0|0.9917664291060544|
|10.0|2.0|5.0|100.0|0.9917664291060544|
|10.0|2.0|2.0|200.0|0.9916778236281828|
|nan|2.0|2.0|200.0|0.9916778236281828|
|nan|2.0|5.0|200.0|0.9914403761900116|

## Top 5 por f1_macro
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_f1_macro|
|---|---|---|---|---|
|nan|1.0|2.0|100.0|0.9508702644829804|
|10.0|1.0|2.0|100.0|0.9508702644829804|
|10.0|1.0|5.0|200.0|0.9444698282761136|
|nan|1.0|5.0|200.0|0.9444698282761136|
|nan|1.0|2.0|200.0|0.9426226579783868|

## Top 5 por accuracy
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_accuracy|
|---|---|---|---|---|
|nan|1.0|2.0|100.0|0.9595744680851064|
|10.0|1.0|2.0|100.0|0.9595744680851064|
|nan|1.0|5.0|200.0|0.953191489361702|
|nan|1.0|2.0|200.0|0.953191489361702|
|10.0|1.0|2.0|200.0|0.953191489361702|

## Mejor configuración
```json
{
  "best_params_": {
    "max_depth": null,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "n_estimators": 100
  },
  "best_score_": 0.9917664291060543,
  "refit_metric": "roc_auc"
}
```
---

# Resumen HPO
Refit metric: neg_root_mean_squared_error

## Top 5 por neg_mean_absolute_error
|param_max_depth|param_min_samples_leaf|param_min_samples_split|param_n_estimators|mean_test_neg_mean_absolute_error|
|---|---|---|---|---|
|10.0|1.0|2.0|100.0|-0.5683471867719218|

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
  "best_score_": -0.5683471867719218,
  "refit_metric": "neg_mean_absolute_error"
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