FROM python:3.12-slim

WORKDIR /app

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
# Browser UI served by the model server at /
COPY web/ /app/web/

# Create directories for transcripts
RUN mkdir -p /app/transcripts

# Set default configuration values
ENV WHISPER_MODEL=large-v3-turbo
ENV OUTPUT_FORMAT=txt
ENV INCLUDE_DIARIZATION=false
ENV FORCE_CPU=true
ENV CACHE_ENABLED=true

# Expose port for server
EXPOSE 8000

# Command to run the server
CMD ["python", "/app/scripts/model_server.py", "--host", "0.0.0.0"]