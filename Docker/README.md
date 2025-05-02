# AI File Manager (Streamlit Docker Deployment)

This document explains how to package your Streamlit-based Python app into a Docker container and run it independently.

## Prerequisites
- Docker installed on your machine.
- Your Streamlit app (`app.py`) ready.
- External APIs (Unstructured OCR, Ollama) already deployed and accessible.

## Files in this Bundle
- `unstructured_chatbot.py`: The Streamlit application code.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Instructions to build the Docker image.
- `docker-compose.yml`: Optional—simplifies running the container.

## Steps to Dockerize and Deploy

1. **Write your Dockerfile** (see below).
2. **Build the image**:
   ```bash
   docker build -t ai-file-manager:latest .


UNSTRUCTURE

docker run -p 9500:9500 -d --rm --name unstructured-api -e PORT=9500 downloads.unstructured.io/unstructured-io/unstructured-api:latest



#unstructured io command that works

http://localhost:9500/general/v0/general
