import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

class GunaClassifier:
    """
    Simple baseline classifier for guna (sattva/rajas/tamas).
    Uses scikit-learn pipeline so it trains fast in Codespaces.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_version = "baseline-0.1"

    def train(self, texts, labels, validation_split: float = 0.2):
        logger.info(f"Training baseline model on {len(texts)} samples")

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Simple bag-of-words + logistic regression
        self.model = make_pipeline(
            CountVectorizer(analyzer="char", ngram_range=(2, 4)),
            LogisticRegression(max_iter=500)
        )

        self.model.fit(texts, y)
        self.is_trained = True

        acc = float(self.model.score(texts, y))
        logger.info(f"Training accuracy (on train set) = {acc:.3f}")

        return {"train_accuracy": acc}

    def predict(self, text: str):
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        proba = self.model.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        guna = self.label_encoder.inverse_transform([idx])[0]
        confidence = float(proba[idx])

        return {
            "input_text": text,
            "predicted_guna": guna,
            "confidence": confidence,
            "all_probabilities": {
                label: float(p) for label, p in zip(self.label_encoder.classes_, proba)
            },
            "emoji": {"sattva": "🟢", "rajas": "🟠", "tamas": "⚫"}.get(guna, "❓"),
            "model_version": self.model_version,
        }

    def get_model_info(self):
        return {
            "is_trained": self.is_trained,
            "model_version": self.model_version,
            "classes": list(self.label_encoder.classes_) if self.is_trained else [],
            "architecture": "sklearn CountVectorizer + LogisticRegression",
        }

    def save(self, path: str):
        logger.info(f"(stub) save() called with path={path}")

    def load(self, path: str):
        logger.info(f"(stub) load() called with path={path}")
