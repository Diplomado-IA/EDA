"""MLPipeline orquestadora del flujo ML"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self, config=None):
        self.config = config
        self.df = None

    def load_data(self):
        logger.info("load_data: noop placeholder")
        return self

    def explore_data(self, output_dir: str | Path | None = None):
        logger.info("explore_data: noop placeholder")
        return {}

    def preprocess_data(self):
        logger.info("preprocess_data: noop placeholder")
        return self

    def engineer_features(self):
        logger.info("engineer_features: noop placeholder")
        return self

    def train_models(self):
        logger.info("train_models: noop placeholder")
        return {}

    def evaluate_models(self):
        logger.info("evaluate_models: noop placeholder")
        return {}

    def run_full_pipeline(self):
        logger.info("run_full_pipeline: start")
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.engineer_features()
        results = {
            "train": self.train_models(),
            "eval": self.evaluate_models(),
        }
        logger.info("run_full_pipeline: end")
        return results
