import json
import os

from training.train import train_and_evaluate


def test_training_runs():
model, metrics = train_and_evaluate(model_version="test-sha")
assert hasattr(model, "predict")
assert "rmse" in metrics and metrics["rmse"] > 0
# sanity: prediction is scalar float
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"]).iloc[[0]]
pred = float(model.predict(X)[0])
assert isinstance(pred, float)