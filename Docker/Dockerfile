FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use JSON array syntax for CMD
CMD ["streamlit", "run", "app.py", "--server.port=8501"]