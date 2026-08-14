# Serving image for the SupportIQ triage API. Installs
# requirements-serving.txt, not the full project requirements.txt — the
# two have a genuine dependency conflict (mlflow pins pandas<3, the dev
# requirements.txt pins pandas==3.0.5 for the training pipeline), so a
# single shared requirements file cannot satisfy both.
FROM python:3.14-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
