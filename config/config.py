from pathlib import Path

class Config:
    DATA_DIR = Path("data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    DATASET_PATH = RAW_DIR / "TITULADO_2007-2024_web_19_05_2025_E.csv"
    OUTPUTS_DIR = Path("outputs")
    EDA_DIR = OUTPUTS_DIR / "eda"
    METRICS_DIR = OUTPUTS_DIR / "metrics"
    MODELS_DIR = Path("models")
    TRAINED_DIR = MODELS_DIR / "trained"
    METADATA_DIR = MODELS_DIR / "metadata"
    REPORTS_DIR = Path("reports")
    TARGET_CLASSIFICATION = "MODALIDAD_BIN"
    TARGET_REGRESSION = "PROMEDIO_EDAD_PROGRAMA"
    METRICS = {
        "classification": ["AUC_PR", "F1_macro"],
        "regression": ["MAE", "RMSE"]
    }
    RISKS = ["desbalance_clases", "nulidad_edad", "drift_post_2020"]
    SEPARATOR = ";"
    ENCODING = "latin1"
    def to_dict(self):
        return {
            "DATASET_PATH": str(self.DATASET_PATH),
            "OUTPUTS_DIR": str(self.OUTPUTS_DIR),
            "TRAINED_DIR": str(self.TRAINED_DIR),
            "METADATA_DIR": str(self.METADATA_DIR),
            "REPORTS_DIR": str(self.REPORTS_DIR),
            "TARGET_CLASSIFICATION": self.TARGET_CLASSIFICATION,
            "TARGET_REGRESSION": self.TARGET_REGRESSION,
        }
