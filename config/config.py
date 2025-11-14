from pathlib import Path

class Config:
    # Paths
    DATA_DIR = Path("data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    DATASET_PATH = RAW_DIR / "TITULADO_2007-2024_web_19_05_2025_E.csv"  # dataset 2007-2024
    OUTPUTS_DIR = Path("outputs")
    EDA_DIR = OUTPUTS_DIR / "eda"
    METRICS_DIR = OUTPUTS_DIR / "metrics"
    MODELS_DIR = Path("models")
    TRAINED_DIR = MODELS_DIR / "trained"
    METADATA_DIR = MODELS_DIR / "metadata"
    REPORTS_DIR = Path("reports")

    # Targets (project proposal)
    TARGET_CLASSIFICATION = "MODALIDAD_BIN"  # Presencial vs No presencial (binarizada)
    TARGET_REGRESSION = "PROMEDIO EDAD PROGRAMA "  # columna original incluye espacio final

    # Metrics (extended per proposal)
    METRICS = {
        "classification": ["AUC_PR", "F1_macro", "Brier"],
        "regression": ["MAE", "MedAE", "RMSE"],
    }

    # Global config
    SEPARATOR = ";"
    ENCODING = "latin1"
    RANDOM_STATE = 42

    # Temporal split strategy (train <=2018, gap 2019, test 2020-2024)
    TRAIN_START_YEAR = 2007
    TRAIN_END_YEAR = 2018
    GAP_YEAR = 2019
    TEST_START_YEAR = 2020
    TEST_END_YEAR = 2024
    TRAIN_TEST_SPLIT = f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR} | {TEST_START_YEAR}-{TEST_END_YEAR} (gap {GAP_YEAR})"

    # Risk registers
    RISKS = [
        "desbalance_modalidad",
        "nulidad_rangos_edad",
        "drift_post_2020",
        "data_leakage",
    ]

    # Imputation strategies (placeholders)
    IMPUTE_NUM = "median"
    IMPUTE_CAT = "most_frequent"
    IMPUTE_RANGOS_EDAD_FILL = 0  # para rangos de edad vacíos

    # Candidate models (tabular per proposal)
    CLASSIFIERS = ["LogReg", "CatBoost", "XGBoost"]
    REGRESSORS = ["ElasticNet", "CatBoost", "LightGBM"]

    # Regularization / training flags
    USE_EARLY_STOPPING = True
    USE_WEIGHT_DECAY = True
    USE_DROPOUT = True

    def to_dict(self):
        return {
            "DATASET_PATH": str(self.DATASET_PATH),
            "TARGET_CLASSIFICATION": self.TARGET_CLASSIFICATION,
            "TARGET_REGRESSION": self.TARGET_REGRESSION,
            "METRICS": self.METRICS,
            "TRAIN_TEST_SPLIT": self.TRAIN_TEST_SPLIT,
            "RANDOM_STATE": self.RANDOM_STATE,
            "RISKS": self.RISKS,
        }
