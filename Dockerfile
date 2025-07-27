FROM python:3.12-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY scripts/ ./scripts/

RUN python scripts/export_e5_to_onnx.py

COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["main.py"]
