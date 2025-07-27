FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  poppler-utils \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install required packages
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

RUN python scripts/export_e5_to_onnx.py

RUN mkdir -p /app/input

# Set environment variables with defaults
ENV INPUT_DIR=/app/input
ENV MAX_DOCUMENTS=10
ENV MAX_TOP_CHUNKS=5
ENV MAX_THREADS=4
ENV MIN_CHUNK_LENGTH=100

# Set the entrypoint
ENTRYPOINT ["python", "main.py"]
