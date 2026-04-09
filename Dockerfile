FROM python:3.11-slim

# System deps for pyarrow / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies (cached layer — only reruns when pyproject.toml changes)
COPY pyproject.toml .
RUN uv sync --no-install-project

# Copy source
COPY src/ ./src/
COPY data/ ./data/

# Streamlit config
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONPATH=/app/src

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/app/wtp_app.py", "--server.address=0.0.0.0"]
