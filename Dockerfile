# Raw2Zarr Docker Image - UV-based
# Radar DataTree framework for FAIR and cloud-native radar data processing
# https://doi.org/10.48550/arXiv.2510.24943

FROM python:3.12-slim

LABEL maintainer="Alfonso Ladino-Rincon <alfonso8@illinois.edu>"
LABEL description="Raw2Zarr - Radar DataTree framework for converting radar data to ARCO Zarr format"
LABEL version="0.5.0"
LABEL org.opencontainers.image.source="https://github.com/aladinor/raw2zarr"
LABEL org.opencontainers.image.licenses="CC-BY-NC-SA-4.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /opt/raw2zarr

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the entire repository
COPY . .

# Install raw2zarr in development mode
RUN uv pip install -e .

# Set environment variables for OSN compatibility
ENV AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Expose JupyterLab port
EXPOSE 8888

# Expose Dask dashboard port
EXPOSE 8787

# Set the default command to launch JupyterLab
CMD ["uv", "run", "jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''"]
