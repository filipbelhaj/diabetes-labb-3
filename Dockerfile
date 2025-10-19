# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1
WORKDIR /app

# Install runtime deps only
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and artifacts (model + metrics)
COPY service/ ./service/
COPY artifacts/ ./artifacts/

# Non-root user (optional)
RUN useradd -m appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]