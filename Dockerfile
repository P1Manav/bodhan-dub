FROM python:3.11-slim

# Install system dependencies (ffmpeg is strictly required for media probe/extract/mux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Copy application source code and configuration
COPY languages.yaml glossary.example.yaml voices.example.yaml ./
COPY src/ ./src/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY Makefile README.md ./

ENTRYPOINT ["python", "-m", "bodhan_dub"]
CMD ["--help"]
