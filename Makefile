.PHONY: venv install train run lint test docker-build docker-run

venv:
python -m venv .venv
. .venv/bin/activate && pip install -r requirements.txt

install:
pip install -r requirements.txt

train:
python training/train.py --out-dir artifacts --model-version local-dev

run:
uvicorn service.app:app --host 0.0.0.0 --port 8000

lint:
ruff check . && black --check .

test:
pytest -q

docker-build:
docker build -t ghcr.io/<org>/<repo>:v0.1 .

docker-run:
docker run --rm -p 8000:8000 ghcr.io/<org>/<repo>:v0.1