# Phase 13: Production AWS Docker Containerization
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# System dependencies for XGBoost/LightGBM/OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code & artifacts
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check endpoint for AWS ECS / App Runner / Load Balancer
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit server
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
