ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm AS main

COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /uvx /bin/

# system libs for scientific wheels / optional geo stack (aligned with main globi visualizer image)
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential ca-certificates \
        libx11-6 libexpat1 libgomp1 libcurl4-openssl-dev libssl-dev libxml2-dev gfortran \
        curl tk python3-tk \
        git \
        gdal-bin libgdal-dev libproj-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

COPY uv.lock pyproject.toml README.md /code/
RUN uv sync --frozen --no-install-project

COPY src /code/src/
COPY .streamlit/config.toml /code/.streamlit/config.toml

RUN uv sync --frozen

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "src/app/wtp_app.py"]
