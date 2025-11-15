"""Definiciones de modelos para Fase 4 (ligeras, sin dependencias pesadas)"""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp_classifier(random_state: int = 42) -> Pipeline:
    """MLP de 2 capas ocultas (128, 64) con early stopping y L2.
    Nota: scikit-learn no soporta BatchNorm/Dropout; usamos early_stopping + L2 como regularización.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,               # L2
        learning_rate="adaptive",
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=random_state,
        verbose=False,
    )
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # seguro para matrices dispersas
        ("mlp", mlp),
    ])


class ModelTrainer:
    def train(self, X, y):
        model = build_mlp_classifier()
        model.fit(X, y)
        return {"mlp_classifier": model}


class ModelEvaluator:
    def evaluate(self, models, X, y):
        from sklearn.metrics import f1_score, average_precision_score
        results = {}
        for name, m in models.items():
            try:
                proba = m.predict_proba(X)[:, 1]
                preds = (proba >= 0.5).astype(int)
                results[name] = {
                    "F1_macro": float(f1_score(y, preds, average="macro")),
                    "AUC_PR": float(average_precision_score(y, proba)),
                }
            except Exception:
                preds = m.predict(X)
                results[name] = {
                    "F1_macro": float(f1_score(y, preds, average="macro")),
                }
        return results
