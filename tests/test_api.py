"""Basic API tests that do not require the external Kaggle dataset."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas import FEATURE_NAMES


class FakeModel:
    """Small deterministic model double for endpoint tests."""

    def predict(self, frame):
        assert list(frame.columns) == list(FEATURE_NAMES)
        return [1]

    def predict_proba(self, frame):
        assert list(frame.columns) == list(FEATURE_NAMES)
        return [[0.18, 0.82]]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    def load_fake_model() -> None:
        main.model_service.model = FakeModel()
        main.model_service.features = list(FEATURE_NAMES)
        main.model_service.model_type = "FakeModel"

    monkeypatch.setattr(main.model_service, "load", load_fake_model)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_info(client: TestClient) -> None:
    response = client.get("/info")

    assert response.status_code == 200
    assert response.json()["model_type"] == "FakeModel"
    assert response.json()["features"] == list(FEATURE_NAMES)


def test_predict(client: TestClient) -> None:
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"heart_disease": True, "probability": 0.82}


def test_predict_rejects_unknown_fields(client: TestClient) -> None:
    response = client.post("/predict", json={"age": 52, "unexpected": 1})

    assert response.status_code == 422
