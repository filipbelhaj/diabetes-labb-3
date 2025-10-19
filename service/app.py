from __future__ import annotations

app = FastAPI(title="Virtual Diabetes Clinic — Progression Risk API", version="0.1")

# Load model at startup
if not MODEL_PATH.exists():
raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Train first.")

model = joblib.load(MODEL_PATH)

# model version resolution
if VERSION_FILE.exists():
model_version = VERSION_FILE.read_text(encoding="utf-8").strip()
else:
# fallback from metrics if needed
try:
with open(METRICS_PATH, "r", encoding="utf-8") as f:
model_version = json.load(f).get("model_version", "unknown")
except Exception:
model_version = "unknown"


@app.get("/health")
async def health():
return {"status": "ok", "model_version": model_version}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: DiabetesFeatures):
try:
X = [[
features.age,
features.sex,
features.bmi,
features.bp,
features.s1,
features.s2,
features.s3,
features.s4,
features.s5,
features.s6,
]]
yhat = float(model.predict(X)[0])
return PredictionResponse(prediction=yhat)
except Exception as e:
# Ensure JSON error
raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def json_exception_handler(request, exc):
# Fallback to guarantee JSON
return JSONResponse(status_code=500, content={"detail": str(exc)})