# Base image
FROM python:3.10-slim

# Install required packages
RUN apt-get update && apt-get install -y \
    curl gnupg apt-transport-https ca-certificates fuse && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt gcsfuse-bullseye main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y gcsfuse && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create the mount point directory for gcsfuse
RUN mkdir -p /app/model

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8000

# Start FastAPI server
# Local run
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# GCP run
CMD bash -c "sleep 5 && gcsfuse $BUCKET_NAME /app/model && uvicorn app:app --host 0.0.0.0 --port 8000"