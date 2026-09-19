"""Load and use the trained model without reloading it per request."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from app.schemas import FEATURE_NAMES


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "heart_model.joblib"


class ModelServiceError(Exception):
    """Base exception for expected model service failures."""


class ModelNotFoundError(ModelServiceError):
    """Raised when training has not produced a model artifact."""


class ModelNotReadyError(ModelServiceError):
    """Raised when a prediction is requested before loading a model."""


class PredictionError(ModelServiceError):
    """Raised when the loaded model cannot make a prediction."""


class ModelLoadError(ModelServiceError):
    """Raised when a model artifact is invalid or cannot be read."""


class ModelService:
    """Own the model lifecycle and enforce the training feature order."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: Any | None = None
        self.features: list[str] = list(FEATURE_NAMES)
        self.model_type: str | None = None

    def load(self) -> None:
        """Load and validate the joblib artifact once during app startup."""

        self.model = None
        self.model_type = None

        if not self.model_path.is_file():
            raise ModelNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Place data/heart.csv in the project and run 'python train.py'."
            )

        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:  # joblib can raise several deserialization errors.
            logger.exception("Unable to load model artifact from %s", self.model_path)
            raise ModelLoadError("The trained model artifact could not be loaded.") from exc

        if not isinstance(artifact, dict):
            raise ModelLoadError("The trained model artifact has an unsupported format.")

        artifact_features = artifact.get("features")
        model = artifact.get("model")
        if list(artifact_features or []) != list(FEATURE_NAMES):
            raise ModelLoadError("The model feature order does not match the API schema.")
        if model is None or not hasattr(model, "predict"):
            raise ModelLoadError("The trained model artifact does not contain a usable model.")

        self.features = list(FEATURE_NAMES)
        self.model = model
        self.model_type = str(artifact.get("model_type") or type(model).__name__)

    @property
    def is_ready(self) -> bool:
        """Return whether a validated model is available for requests."""

        return self.model is not None

    def info(self) -> dict[str, Any]:
        """Return safe, public model metadata."""

        self._ensure_ready()
        return {
            "model_type": self.model_type or type(self.model).__name__,
            "features": list(self.features),
        }

    def predict(self, values: Mapping[str, Any]) -> tuple[bool, float]:
        """Predict the label and positive-class probability for one patient."""

        self._ensure_ready()

        try:
            input_frame = pd.DataFrame(
                [[values[feature] for feature in self.features]],
                columns=self.features,
            )
            prediction = int(self.model.predict(input_frame)[0])
            if prediction not in (0, 1):
                raise ValueError("The model returned a non-binary prediction.")

            if not hasattr(self.model, "predict_proba"):
                raise ValueError("The model does not provide prediction probabilities.")
            probability = float(self.model.predict_proba(input_frame)[0][1])
            return bool(prediction), round(probability, 4)
        except Exception as exc:
            raise PredictionError("The model could not process this prediction.") from exc

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise ModelNotReadyError(
                "The trained model is not available. Run 'python train.py' before predicting."
            )
