import requests
from fastapi.testclient import TestClient
from ml_project.online_inference.app import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hi=)"}


def test_predict():
    json = {
        "data": [
            [69.0, 1.0, 0.0, 160.0, 234.0, 1.0, 2.0, 131.0, 0.0, 0.1, 1.0, 1.0, 0.0],
            [68.0, 0.0, 1.0, 140.0, 239.0, 0.0, 0.0, 151.0, 1.0, 1.8, 0.0, 2.0, 1.0]
        ],
        "columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"
        ]
    }
    with TestClient(app) as client:
        response = client.get("/predict", json=json)
    assert response.status_code == 200
    assert response.json() == [{'prod': 1.0}, {'prod': 0.0}]