# --- Stage 1: Build & Train ---
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/raw/diabetic_data.csv ./data/raw/diabetic_data.csv

# Run the pipeline to generate data splits, features, and train the model
# This ensures the model artifact is baked into the image
RUN python -m src.data.preprocessor
RUN python -m src.features.build_features
RUN python -m src.models.train

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the app code and the trained artifacts from the builder
COPY --from=builder /app/src ./src
COPY --from=builder /app/models ./models
COPY --from=builder /app/config ./config

# Expose the API port
EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]