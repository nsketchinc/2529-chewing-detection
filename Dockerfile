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
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main_websocket.py .
COPY util/ ./util/

# Copy ML models and data
COPY data/tmp_model/ ./data/tmp_model/
COPY data/face_landmarker.task ./data/face_landmarker.task

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the application with eventlet worker
CMD ["python", "main_websocket.py"]
