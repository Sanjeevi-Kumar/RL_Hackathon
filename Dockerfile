FROM python:3.11-slim

LABEL maintainer="Warehouse RL Team"
LABEL description="OpenEnv-compatible Warehouse RL Environment"
LABEL version="1.0.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY inference.py .

# Expose the OpenEnv HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server
CMD ["uvicorn", "src.envs.warehouse_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
