# Use Python 3.11 slim to match your conda environment
FROM python:3.11-slim

# Set working directory
WORKDIR /weather

# Install system dependencies needed for scientific packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and setuptools to latest versions (critical for TF 2.18.0)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /weather
USER app

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; import os; requests.get(f'http://0.0.0.0:{os.environ.get(\"PORT\", 5000)}/health')" || exit 1


# Run Flask app
CMD ["python", "src/app.py"]