"""FastAPI application for heart disease predictions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from app.model_service import ModelService, ModelServiceError, PredictionError
from app.schemas import (
    HeartDiseaseInput,
    HealthResponse,
    InfoResponse,
    PredictionResponse,
)


logger = logging.getLogger(__name__)
API_VERSION = "1.0.0"
model_service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model once while starting the application."""

    try:
        model_service.load()
        logger.info("Loaded %s model from %s", model_service.model_type, model_service.model_path)
    except ModelServiceError as exc:
        # Keep the process alive so health checks return a useful 503 response.
        logger.error("Model service is unavailable: %s", exc)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="An educational binary classification API for heart disease risk prediction.",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report healthy only when the application and model are ready."""

    if not model_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: the trained model is not loaded.",
        )
    return HealthResponse(status="healthy")


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Return the model type and exact feature order used by predictions."""

    try:
        model_info = model_service.info()
    except ModelServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return InfoResponse(**model_info, api_version=API_VERSION)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: HeartDiseaseInput) -> PredictionResponse:
    """Return a binary prediction and positive-class probability."""

    try:
        heart_disease, probability = model_service.predict(patient.model_dump())
    except ModelServiceError as exc:
        status_code = 503 if not model_service.is_ready else 500
        if isinstance(exc, PredictionError):
            logger.exception("Prediction failed")
            detail = "Prediction failed while processing the supplied patient data."
        else:
            detail = str(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc

    return PredictionResponse(heart_disease=heart_disease, probability=probability)
