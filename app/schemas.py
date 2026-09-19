"""Pydantic request and response schemas for the API."""

from pydantic import BaseModel, ConfigDict, Field


FEATURE_NAMES: tuple[str, ...] = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)


class HeartDiseaseInput(BaseModel):
    """Patient measurements accepted by the prediction endpoint."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="Dataset sex encoding")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type encoding")
    trestbps: float = Field(..., ge=0, le=400, description="Resting blood pressure")
    chol: float = Field(..., ge=0, le=1000, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar encoding")
    restecg: int = Field(..., ge=0, le=3, description="Resting ECG encoding")
    thalach: float = Field(..., ge=0, le=300, description="Maximum heart rate")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina encoding")
    oldpeak: float = Field(..., ge=-20, le=20, description="ST depression")
    slope: int = Field(..., ge=0, le=3, description="Slope encoding")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=4, description="Thalassemia encoding")


class HealthResponse(BaseModel):
    """Response returned when the service is ready."""

    status: str


class InfoResponse(BaseModel):
    """Public model metadata."""

    model_type: str
    features: list[str]
    api_version: str


class PredictionResponse(BaseModel):
    """Prediction returned to API clients."""

    heart_disease: bool
    probability: float
