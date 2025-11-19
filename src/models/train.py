"""Utilidades de entrenamiento y GridSearchCV."""
from typing import Any, Dict
import os
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV


def run_grid_search(estimator, param_grid: Dict[str, Any], X, y, cv, scoring=("neg_root_mean_squared_error"), n_jobs: int = -1, verbose: int = 0, out_dir: str = "outputs/hpo"):
    os.makedirs(out_dir, exist_ok=True)
    refit_metric = scoring[0] if isinstance(scoring, (list, tuple)) else scoring
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit_metric,
        return_train_score=True,
    )
    gs.fit(X, y)
    pd.DataFrame(gs.cv_results_).to_csv(os.path.join(out_dir, "results.csv"), index=False)
    best = {"best_params_": gs.best_params_, "best_score_": gs.best_score_, "refit_metric": refit_metric}
    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    return gs
