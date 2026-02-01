# Use Python 3.11 (3.14 not widely available in Docker images yet)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main_websocket.py .
COPY util/ ./util/

# Copy ML models and data (ensure directory structure is preserved)
COPY data/tmp_model/exp001/ ./data/tmp_model/exp001/
COPY data/face_landmarker.task ./data/face_landmarker.task

# Debug: Verify model files are copied
RUN echo "=== Checking model directory ===" && \
    ls -lah data/tmp_model/exp001/ && \
    echo "=== feat_cols.pickle exists? ===" && \
    test -f data/tmp_model/exp001/feat_cols.pickle && echo "YES" || echo "NO"

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the application with eventlet worker
CMD ["python", "main_websocket.py"]
