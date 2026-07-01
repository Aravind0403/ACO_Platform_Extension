FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY extender/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core modules (research repo — unchanged)
COPY core/ ./core/

# Copy extender source
COPY extender/ ./extender/

# Make core importable
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "extender.main:app", "--host", "0.0.0.0", "--port", "8080"]
