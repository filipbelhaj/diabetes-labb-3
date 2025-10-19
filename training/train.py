from __future__ import annotations
pipeline = Pipeline(
steps=[
("scaler", StandardScaler()),
("reg", LinearRegression()),
]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Simple latency probe (ms) over single predictions
import time

samples = X_test.iloc[:100]
latencies = []
for _, row in samples.iterrows():
x = row.values.reshape(1, -1)
t0 = time.perf_counter()
_ = pipeline.predict(x)
latencies.append((time.perf_counter() - t0) * 1000)

p50 = float(np.percentile(latencies, 50))
p95 = float(np.percentile(latencies, 95))

metrics = {
"model_version": model_version,
"trained_at": datetime.now(timezone.utc).isoformat(),
"random_state": RANDOM_STATE,
"test_size": TEST_SIZE,
"n_features": X.shape[1],
"n_train": int(len(X_train)),
"n_test": int(len(X_test)),
"rmse": float(rmse),
"latency_ms_p50": p50,
"latency_ms_p95": p95,
# image_size_mb will be filled in by release workflow
}

return pipeline, metrics


def main():
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="artifacts", help="Output directory")
parser.add_argument("--model-version", default="dev", help="Model version identifier")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

model, metrics = train_and_evaluate(args.model_version)

model_path = os.path.join(args.out_dir, "model.pkl")
joblib.dump(model, model_path)

metrics_path = os.path.join(args.out_dir, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
json.dump(metrics, f, indent=2)

print(f"Saved model to {model_path}")
print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
main()
