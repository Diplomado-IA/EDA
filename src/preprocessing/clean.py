"""Preprocesamiento Fase 3: limpieza, split temporal, codificación y features.
Todos los pasos generan artefactos y reportes en data/processed y outputs/eda.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression
import logging
logger = logging.getLogger(__name__)

from config.config import Config

PROC_DIR = Path("data/processed")
EDA_DIR = Path("outputs/eda")
PROC_DIR.mkdir(parents=True, exist_ok=True)
(EDA_DIR / "resumen").mkdir(parents=True, exist_ok=True)
(EDA_DIR / "figures").mkdir(parents=True, exist_ok=True)


def _ensure_modalidad_bin(df: pd.DataFrame) -> pd.DataFrame:
    if Config.TARGET_CLASSIFICATION in df.columns:
        return df
    if "MODALIDAD" in df.columns:
        tmp = df["MODALIDAD"].astype(str).str.strip().str.lower()
        df["MODALIDAD_BIN"] = tmp.apply(lambda v: 1 if v.startswith("presencial") else 0)
    return df


def _coerce_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    col = Config.TARGET_REGRESSION
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(num_cols):
        if Config.IMPUTE_NUM == "median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif Config.IMPUTE_NUM == "mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    if len(cat_cols):
        if Config.IMPUTE_CAT == "most_frequent":
            modes = df[cat_cols].mode(dropna=True).iloc[0]
            df[cat_cols] = df[cat_cols].fillna(modes)
        elif Config.IMPUTE_CAT == "constant":
            df[cat_cols] = df[cat_cols].fillna("missing")
    return df


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal robusto.
    Extrae el año numérico desde la columna 'AÑO' (valores como 'TIT_2024') y realiza el split.
    """
    year_col = None
    for c in df.columns:
        if c.strip().upper() == "AÑO":
            year_col = c
            break
    if year_col is None:
        raise KeyError("No se encontró columna 'AÑO' (ni variantes con espacios)")
    # Log de dtype y ejemplos
    try:
        logger.info(f"dtype AÑO: {df[year_col].dtype}; sample: {df[year_col].head().tolist()}")
    except Exception:
        logger.info("No se pudo loggear muestra de 'AÑO'")
    # Extraer los últimos cuatro dígitos (año) de cada valor
    year_series = df[year_col].astype(str).str.extract(r"(\d{4})", expand=False)
    df_valid = df.copy()
    # Coerción robusta a número, rellena NaN con -1 y fuerza dtype int
    df_valid["__AÑO_NUM__"] = pd.to_numeric(year_series, errors="coerce")
    df_valid["__AÑO_NUM__"] = df_valid["__AÑO_NUM__"].fillna(-1).astype(int)
    # Si por alguna razón quedó object, volver a convertir
    if df_valid["__AÑO_NUM__"].dtype == "O":
        df_valid["__AÑO_NUM__"] = pd.to_numeric(df_valid["__AÑO_NUM__"], errors="coerce").fillna(-1).astype(int)
    # Máscara usando between (evita comparaciones str-int)
    # Forzar años de config a int por si fueron cargados como str
    train_start = int(Config.TRAIN_START_YEAR)
    train_end = int(Config.TRAIN_END_YEAR)
    test_start = int(Config.TEST_START_YEAR)
    test_end = int(Config.TEST_END_YEAR)
    # Re-coerción final de la columna numérica
    df_valid["__AÑO_NUM__"] = pd.to_numeric(df_valid["__AÑO_NUM__"], errors="coerce").fillna(-1).astype(int)
    # Aplicar máscaras (debug previo)
    logger.info(f"Antes de between: dtype={df_valid['__AÑO_NUM__'].dtype}; sample={df_valid['__AÑO_NUM__'].head().tolist()}")
    try:
        train_mask = df_valid["__AÑO_NUM__"].between(train_start, train_end)
        test_mask = df_valid["__AÑO_NUM__"].between(test_start, test_end)
    except Exception as err:
        logger.error(f"Error comparando años: dtype={df_valid['__AÑO_NUM__'].dtype}, sample={df_valid['__AÑO_NUM__'].head().tolist()} -> {err}")
        import traceback
        tb = traceback.format_exc()
        raise TypeError(f"Fallo between años. Config: train({train_start},{train_end}) test({test_start},{test_end}) dtype={df_valid['__AÑO_NUM__'].dtype} sample={df_valid['__AÑO_NUM__'].head().tolist()} error={err} trace={tb}")
    train_df = df_valid.loc[train_mask & (df_valid["__AÑO_NUM__"] != -1)].copy()
    test_df = df_valid.loc[test_mask & (df_valid["__AÑO_NUM__"] != -1)].copy()
    # Si ambos vacíos, lanzar aviso
    if train_df.empty and test_df.empty:
        raise ValueError("No se pudieron extraer años válidos de la columna 'AÑO'. Ejemplos: "
                         + ", ".join(df[year_col].astype(str).head(5).tolist()))
    # Actualizar columna año a numérico si procede
    try:
        train_df[year_col] = train_df["__AÑO_NUM__"].astype(int)
        test_df[year_col] = test_df["__AÑO_NUM__"].astype(int)
    except Exception:
        pass
    train_df.drop(columns=["__AÑO_NUM__"], inplace=True)
    test_df.drop(columns=["__AÑO_NUM__"], inplace=True)
    return train_df, test_df


def scale_numeric(train: pd.DataFrame, test: pd.DataFrame, exclude: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    exclude = set(exclude or [])
    num_cols = [c for c in train.select_dtypes(include=["number"]).columns if c not in exclude]
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    if num_cols:
        scaler.fit(train[num_cols])
        train_scaled[num_cols] = scaler.transform(train[num_cols])
        test_scaled[num_cols] = scaler.transform(test[num_cols])
    details = {"num_cols_scaled": num_cols, "scaler_means": scaler.mean_.tolist() if num_cols else [], "scaler_vars": scaler.var_.tolist() if num_cols else []}
    return train_scaled, test_scaled, details


def one_hot_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Optional[List[str]] = None,
    rare_thresh: float = 0.005,
    high_card_thresh: int = 100,
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Memory-safe encoding for categoricals.
    - Groups rare categories (<rare_thresh in train) into "__RARE__".
    - Uses frequency encoding for high-cardinality columns (>high_card_thresh unique after grouping).
    - Applies one-hot encoding only to remaining categoricals.
    Returns encoded dataframes and the list of processed categorical columns.
    """
    train_out = train.copy()
    test_out = test.copy()

    # Infer candidate categorical columns
    if cols is None:
        cols = train_out.select_dtypes(include=["object"]).columns.tolist()

    # Exclude targets from encoding
    exclude_targets = {Config.TARGET_CLASSIFICATION, Config.TARGET_REGRESSION}
    cols = [c for c in cols if c in train_out.columns and c not in exclude_targets]

    used_cols: List[str] = []

    for col in cols:
        # Work on string view for consistent grouping
        tr = train_out[col].astype(str).fillna("")
        te = test_out[col].astype(str).fillna("")

        # Group rare categories according to train frequency
        freq = tr.value_counts(normalize=True)
        rare_vals = set(freq[freq < rare_thresh].index)
        if rare_vals:
            tr = tr.where(~tr.isin(rare_vals), "__RARE__")
            te = te.where(~te.isin(rare_vals), "__RARE__")

        n_unique = int(tr.nunique(dropna=False))

        if n_unique > high_card_thresh:
            # Frequency encoding
            f = tr.value_counts(normalize=True)
            train_out[f"{col}_freq"] = tr.map(f).astype("float32")
            test_out[f"{col}_freq"] = te.map(f).fillna(0.0).astype("float32")
            # Drop original column
            train_out.drop(columns=[col], inplace=True)
            test_out.drop(columns=[col], inplace=True)
            used_cols.append(col)
        else:
            # One-Hot on this column only, align test to train dummies
            dtr = pd.get_dummies(tr, prefix=col, drop_first=drop_first)
            dte = pd.get_dummies(te, prefix=col, drop_first=drop_first)
            dtr = dtr.astype("uint8")
            dte = dte.astype("uint8")
            dte = dte.reindex(columns=dtr.columns, fill_value=0)

            train_out = pd.concat([train_out.drop(columns=[col]), dtr], axis=1)
            test_out = pd.concat([test_out.drop(columns=[col]), dte], axis=1)
            used_cols.append(col)

    return train_out, test_out, used_cols


def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    report_lines = []
    train_fe = train.copy()
    test_fe = test.copy()

    # Dummies específicos si existen
    for col in ["PLAN", "JORNADA"]:
        if col in train_fe.columns:
            train_fe = pd.get_dummies(train_fe, columns=[col], prefix=col, drop_first=True)
            test_fe = pd.get_dummies(test_fe, columns=[col], prefix=col, drop_first=True)
            test_fe = test_fe.reindex(columns=train_fe.columns, fill_value=0)
            report_lines.append(f"Dummies generados para {col}")

    # Flag POST_2020
    if "AÑO" in train_fe.columns:
        try:
            y_tr = pd.to_numeric(train_fe["AÑO"], errors="coerce")
            y_te = pd.to_numeric(test_fe["AÑO"], errors="coerce")
            train_fe["POST_2020"] = (y_tr >= 2020).astype("uint8")
            test_fe["POST_2020"] = (y_te >= 2020).astype("uint8")
            report_lines.append("Flag POST_2020 creado")
        except Exception:
            pass

    # Ratios simples (seguros)
    def _safe_ratio(df: pd.DataFrame, num: str, den: str, name: str) -> bool:
        if num in df.columns and den in df.columns:
            eps = 1e-6
            df[name] = df[num] / (df[den].replace(0, np.nan) + eps)
            return True
        return False

    ratio_created = False
    if _safe_ratio(train_fe, "TOTAL TITULACIONES", "DURACIÓN ESTUDIO CARRERA", "RATIO_TITULACIONES_DURACION"):
        _safe_ratio(test_fe, "TOTAL TITULACIONES", "DURACIÓN ESTUDIO CARRERA", "RATIO_TITULACIONES_DURACION")
        ratio_created = True
    if _safe_ratio(train_fe, "PROMEDIO_EDAD_PROGRAMA", "DURACIÓN ESTUDIO CARRERA", "RATIO_EDAD_DURACION"):
        _safe_ratio(test_fe, "PROMEDIO_EDAD_PROGRAMA", "DURACIÓN ESTUDIO CARRERA", "RATIO_EDAD_DURACION")
        ratio_created = True
    if ratio_created:
        report_lines.append("Ratios creados (p.ej., RATIO_TITULACIONES_DURACION)")

    # Features temporales (lags/rolling) por institución (si existe) y año
    value_col = "TOTAL TITULACIONES"
    group_candidates = [
        "CÓDIGO INSTITUCIÓN", "CODIGO INSTITUCION", "INSTITUCIÓN", "INSTITUCION", "PROGRAMA"
    ]
    group_col = next((c for c in group_candidates if c in train_fe.columns), None)
    if group_col and value_col in train_fe.columns and "AÑO" in train_fe.columns:
        for df_fe in (train_fe, test_fe):
            df_fe.sort_values([group_col, "AÑO"], inplace=True)
            df_fe[f"{value_col}_lag1"] = df_fe.groupby(group_col)[value_col].shift(1)
            df_fe[f"{value_col}_lag2"] = df_fe.groupby(group_col)[value_col].shift(2)
            df_fe[f"{value_col}_roll3_mean"] = df_fe.groupby(group_col)[value_col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        report_lines.append(f"Lags (1,2) y rolling(3) para {value_col} por {group_col} generados")

    # Agregaciones por REGIÓN y AÑO (sum/mean)
    if {"REGIÓN", "AÑO", value_col}.issubset(train_fe.columns):
        for df_fe in (train_fe, test_fe):
            gb = df_fe.groupby(["REGIÓN", "AÑO"])[value_col]
            df_fe[f"{value_col}_sum_region_year"] = gb.transform("sum")
            df_fe[f"{value_col}_mean_region_year"] = gb.transform("mean")
        report_lines.append("Agregaciones sum/mean por REGIÓN y AÑO generadas")

    # Índices agregados simples (placeholder genérico)
    # HHI: Herfindahl-Hirschman Index sobre distribución de PROGRAMA por AÑO (si existen)
    if {"PROGRAMA", "AÑO"}.issubset(train.columns):
        prog_share = train["PROGRAMA"].value_counts(normalize=True)
        hhi = float((prog_share ** 2).sum())
        train_fe["HHI_GLOBAL"] = hhi
        test_fe["HHI_GLOBAL"] = hhi
        report_lines.append(f"HHI_GLOBAL={hhi:.4f} calculado sobre PROGRAMA")

    # LQ (Location Quotient) simple: proporción regional vs global si REGION existe
    if {"REGIÓN", "PROGRAMA"}.issubset(train.columns):
        global_prog = train["PROGRAMA"].value_counts(normalize=True)
        region_prog = train.groupby("REGIÓN")["PROGRAMA"].value_counts(normalize=True)
        # Crear una característica promedio LQ por fila usando región de la fila
        def lq_row(r):
            try:
                p_reg = region_prog[(r["REGIÓN"], r["PROGRAMA"])]
                p_glob = global_prog[r["PROGRAMA"]]
                return p_reg / p_glob if p_glob > 0 else 1.0
            except Exception:
                return 1.0
        train_fe["LQ_PROGRAMA"] = train.apply(lq_row, axis=1)
        test_fe["LQ_PROGRAMA"] = test.apply(lq_row, axis=1)
        report_lines.append("LQ_PROGRAMA calculado (REGIÓN vs global)")

    # IPG (placeholder): Índice de Paridad de Género si columnas existen
    if {"PROMEDIO_EDAD_HOMBRE", "PROMEDIO_EDAD_MUJER"}.issubset(train.columns):
        eps = 1e-6
        train_fe["IPG_EDAD"] = (train["PROMEDIO_EDAD_MUJER"] + eps) / (train["PROMEDIO_EDAD_HOMBRE"] + eps)
        test_fe["IPG_EDAD"] = (test["PROMEDIO_EDAD_MUJER"] + eps) / (test["PROMEDIO_EDAD_HOMBRE"] + eps)
        report_lines.append("IPG_EDAD generado (Mujer/Hombre)")

    report = "\n".join(report_lines) if report_lines else "Sin features específicas generadas (faltan columnas esperadas)."
    return train_fe, test_fe, report


def correlation_matrix(X: pd.DataFrame, sample_rows: int = 20000, max_cols: int = 200) -> pd.DataFrame:
    X_num = X.select_dtypes(include=["number"]).copy()
    # Cap number of columns by top variance
    if X_num.shape[1] > max_cols:
        variances = X_num.var(numeric_only=True)
        keep_cols = variances.sort_values(ascending=False).head(max_cols).index
        X_num = X_num[keep_cols]
    # Sample rows for efficiency
    if len(X_num) > sample_rows:
        X_num = X_num.sample(n=sample_rows, random_state=Config.RANDOM_STATE)
    # Downcast to float32 to reduce memory/CPU
    X_num = X_num.astype("float32", errors="ignore")
    corr = X_num.corr()
    corr.to_csv(PROC_DIR / "correlation_matrix.csv")
    # Traceability of columns used
    try:
        (PROC_DIR / "correlation_columns_used.txt").write_text("\n".join(X_num.columns), encoding="utf-8")
    except Exception:
        pass
    return corr


def compute_vif(
    X: pd.DataFrame,
    sample_rows: int = 20000,
    corr_thresh: float = 0.90,
    var_thresh: float = 1e-5,
    max_cols: int = 200,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute VIF efficiently using correlation matrix inversion.
    Steps: numeric subset -> drop targets/excludes -> clean -> sample -> drop near-constant -> cap columns by variance -> prune high-corr -> VIF = diag(inv(corr)).
    """
    X_num = X.select_dtypes(include=["number"]).copy()
    # Drop targets and optional excludes
    exclude_set = set(exclude or [])
    exclude_set.update([Config.TARGET_CLASSIFICATION, Config.TARGET_REGRESSION])
    X_num = X_num.drop(columns=[c for c in exclude_set if c in X_num.columns], errors="ignore")
    # Clean values and downcast
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32", errors="ignore")
    # Sample rows for speed
    if len(X_num) > sample_rows:
        X_num = X_num.sample(n=sample_rows, random_state=Config.RANDOM_STATE)
    # Drop near-zero variance columns
    variances = X_num.var(numeric_only=True)
    X_num = X_num.loc[:, variances > var_thresh]
    # Cap number of columns by top variance
    if X_num.shape[1] > max_cols:
        keep_cols = variances.sort_values(ascending=False).head(max_cols).index
        keep_cols = [c for c in keep_cols if c in X_num.columns]
        X_num = X_num[keep_cols]
    # Drop one of highly correlated pairs to stabilize inversion
    if X_num.shape[1] > 1:
        corr = X_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        if to_drop:
            X_num = X_num.drop(columns=to_drop, errors="ignore")
    if X_num.shape[1] == 0:
        vif_df = pd.DataFrame({"feature": [], "vif": []})
        vif_df.to_csv(PROC_DIR / "vif_scores.csv", index=False)
        return vif_df
    # Correlation matrix inversion method
    R = X_num.corr().values
    # Use pseudo-inverse to handle near-singularity
    R_inv = np.linalg.pinv(R)
    vif_vals = np.diag(R_inv)
    vif_df = pd.DataFrame({"feature": X_num.columns, "vif": vif_vals}).sort_values("vif", ascending=False)
    vif_df.to_csv(PROC_DIR / "vif_scores.csv", index=False)
    # Traceability of columns used
    try:
        (PROC_DIR / "vif_columns_used.txt").write_text("\n".join(X_num.columns), encoding="utf-8")
    except Exception:
        pass
    return vif_df


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    top_k: int = 30,
    sample_rows: int = 20000,
    max_cols: int = 500,
    var_thresh: float = 1e-6,
    method: str = "auto",
) -> List[str]:
    """Fast feature selection with sampling and variance cap.
    - Samples rows (up to sample_rows), drops near-constant vars and caps to max_cols by variance.
    - Uses f_classif/f_regression when task-appropriate (much faster); falls back to mutual information.
    """
    X_num = X.select_dtypes(include=["number"]).copy()
    # Clean and downcast
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32", errors="ignore")
    # Sample rows
    if len(X_num) > sample_rows:
        idx = X_num.sample(n=sample_rows, random_state=Config.RANDOM_STATE).index
        X_num = X_num.loc[idx]
        y_s = y.loc[idx] if hasattr(y, "loc") else y.iloc[idx]
    else:
        y_s = y
    # Drop near-zero variance
    variances = X_num.var(numeric_only=True)
    X_num = X_num.loc[:, variances > var_thresh]
    # Cap number of columns by top variance
    if X_num.shape[1] > max_cols:
        keep_cols = variances.sort_values(ascending=False).head(max_cols).index
        keep_cols = [c for c in keep_cols if c in X_num.columns]
        X_num = X_num[keep_cols]
    # Scoring
    try:
        if method == "auto":
            if task == "classification":
                scores, _ = f_classif(X_num, y_s)
            else:
                from sklearn.feature_selection import f_regression as _freg
                scores, _ = _freg(X_num, y_s)
        elif method == "mi":
            if task == "classification":
                scores = mutual_info_classif(X_num, y_s, discrete_features=False, random_state=Config.RANDOM_STATE)
            else:
                scores = mutual_info_regression(X_num, y_s, random_state=Config.RANDOM_STATE)
        else:
            scores, _ = f_classif(X_num, y_s) if task == "classification" else f_regression(X_num, y_s)
    except Exception:
        # Fallback to MI if ANOVA fails
        if task == "classification":
            scores = mutual_info_classif(X_num, y_s, discrete_features=False, random_state=Config.RANDOM_STATE)
        else:
            scores = mutual_info_regression(X_num, y_s, random_state=Config.RANDOM_STATE)
    score_name = "score"
    df_scores = pd.DataFrame({"feature": X_num.columns, score_name: scores}).sort_values(score_name, ascending=False)
    selected = df_scores.head(min(top_k, len(df_scores))).feature.tolist()
    (PROC_DIR / "selected_features.txt").write_text("\n".join(selected), encoding="utf-8")
    return selected


def save_datasets(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    X_train.to_csv(PROC_DIR / "X_train_engineered.csv", index=False)
    X_test.to_csv(PROC_DIR / "X_test_engineered.csv", index=False)


def preprocess_pipeline(df: pd.DataFrame) -> Dict[str, str]:
    """Ejecuta la fase 3 completa y devuelve rutas de artefactos y reportes."""
    artifacts: Dict[str, str] = {}
    # Limpieza
    df = _ensure_modalidad_bin(df)
    df = _coerce_regression_target(df)
    df = impute_values(df)
    # Split
    train_df, test_df = temporal_split(df)
    # Escalado (excluir objetivos)
    exclude = [Config.TARGET_CLASSIFICATION, Config.TARGET_REGRESSION]
    train_df, test_df, scaler_info = scale_numeric(train_df, test_df, exclude=exclude)
    artifacts["scaler_info"] = EDA_DIR.joinpath("resumen/scaler_info.txt").as_posix()
    Path(artifacts["scaler_info"]).write_text(str(scaler_info), encoding="utf-8")
    # Codificación
    train_enc, test_enc, used_cols = one_hot_encode(train_df, test_df)
    artifacts["one_hot_cols"] = EDA_DIR.joinpath("resumen/one_hot_columns.txt").as_posix()
    Path(artifacts["one_hot_cols"]).write_text("\n".join(used_cols), encoding="utf-8")
    # Features
    train_fe, test_fe, fe_report = engineer_features(train_enc, test_enc)
    artifacts["feature_engineering_report"] = PROC_DIR.joinpath("feature_engineering_report.txt").as_posix()
    Path(artifacts["feature_engineering_report"]).write_text(fe_report, encoding="utf-8")
    # Correlación y VIF
    corr = correlation_matrix(train_fe)
    vif = compute_vif(train_fe)
    artifacts["correlation_matrix"] = PROC_DIR.joinpath("correlation_matrix.csv").as_posix()
    artifacts["vif_scores"] = PROC_DIR.joinpath("vif_scores.csv").as_posix()
    # Selección de features (clasificación por defecto si existe Y binaria)
    task = "classification" if Config.TARGET_CLASSIFICATION in train_fe.columns else "regression"
    y_col = Config.TARGET_CLASSIFICATION if task == "classification" else Config.TARGET_REGRESSION
    y_train = train_fe[y_col] if y_col in train_fe.columns else pd.Series(np.zeros(len(train_fe)))
    X_train = train_fe.drop(columns=[c for c in [Config.TARGET_CLASSIFICATION, Config.TARGET_REGRESSION] if c in train_fe.columns])
    X_test = test_fe.drop(columns=[c for c in [Config.TARGET_CLASSIFICATION, Config.TARGET_REGRESSION] if c in test_fe.columns])
    selected = select_features(X_train, y_train, task=task)
    artifacts["selected_features"] = PROC_DIR.joinpath("selected_features.txt").as_posix()
    # Guardar datasets finales
    save_datasets(X_train[X_train.columns.intersection(selected)], X_test[X_test.columns.intersection(selected)])
    artifacts["X_train_engineered"] = PROC_DIR.joinpath("X_train_engineered.csv").as_posix()
    artifacts["X_test_engineered"] = PROC_DIR.joinpath("X_test_engineered.csv").as_posix()
    return artifacts
