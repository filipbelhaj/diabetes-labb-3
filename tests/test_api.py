import json
import os

import joblib
import pytest
from fastapi.testclient import TestClient

from service.app import app, model

client = TestClient(app)


def test_health():
r = client.get("/health")
assert r.status_code == 200
body = r.json()
assert body["status"] == "ok"
assert "model_version" in body


def test_predict_ok():
payload = {
"age": 0.02,
"sex": -0.044,
"bmi": 0.06,
"bp": -0.03,
"s1": -0.02,
"s2": 0.03,
"s3": -0.02,
"s4": 0.02,
"s5": 0.02,
"s6": -0.001,
}
r = client.post("/predict", json=payload)
assert r.status_code == 200
body = r.json()
assert "prediction" in body
assert isinstance(body["prediction"], float)


def test_predict_missing_field():
payload = {"age": 0.01} # invalid
r = client.post("/predict", json=payload)
assert r.status_code == 422