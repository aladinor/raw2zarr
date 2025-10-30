# Docker Usage Guide

This guide explains how to build and run Raw2Zarr using Docker.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up

# Access JupyterLab at http://localhost:8888
# Access Dask dashboard at http://localhost:8787
```

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t raw2zarr:latest .

# Run the container
docker run -p 8888:8888 -p 8787:8787 raw2zarr:latest
```

## Building the Image

### Basic build
```bash
docker build -t raw2zarr:latest .
```

### Build with specific tag
```bash
docker build -t raw2zarr:0.5.0 .
```

### Build for multi-platform (ARM64 + AMD64)
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t raw2zarr:latest .
```

## Running the Container

### Interactive JupyterLab
```bash
docker run -p 8888:8888 -p 8787:8787 \
  -v $(pwd)/notebooks:/opt/raw2zarr/notebooks \
  -v $(pwd)/data:/opt/raw2zarr/data \
  raw2zarr:latest
```

### Run Python script
```bash
docker run -v $(pwd)/data:/data raw2zarr:latest \
  uv run python your_script.py
```

### Interactive shell
```bash
docker run -it raw2zarr:latest bash
```

## Volume Mounts

Mount directories to persist data and access local files:

```bash
docker run \
  -v $(pwd)/notebooks:/opt/raw2zarr/notebooks \
  -v $(pwd)/data:/opt/raw2zarr/data \
  -v $(pwd)/zarr:/opt/raw2zarr/zarr \
  -v ~/.aws:/root/.aws:ro \
  -p 8888:8888 \
  raw2zarr:latest
```

## Environment Variables

Set environment variables for configuration:

```bash
docker run \
  -e AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -p 8888:8888 \
  raw2zarr:latest
```

## Using with AWS/OSN Credentials

### Option 1: Mount AWS credentials directory
```bash
docker run \
  -v ~/.aws:/root/.aws:ro \
  -p 8888:8888 \
  raw2zarr:latest
```

### Option 2: Pass credentials as environment variables
```bash
docker run \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -p 8888:8888 \
  raw2zarr:latest
```

## Docker Compose Examples

### Basic usage
```bash
docker-compose up
```

### Run in background
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f
```

### Stop and remove
```bash
docker-compose down
```

### Rebuild after code changes
```bash
docker-compose up --build
```

## Advanced Usage

### Run with custom command
```bash
docker run raw2zarr:latest uv run python -c "import raw2zarr; print(raw2zarr.__version__)"
```

### Run pytest inside container
```bash
docker run raw2zarr:latest uv run pytest tests/
```

### Run with Dask cluster
```bash
# Start scheduler
docker run -p 8786:8786 -p 8787:8787 raw2zarr:latest \
  uv run dask scheduler

# Start workers (in separate terminals)
docker run --network host raw2zarr:latest \
  uv run dask worker tcp://localhost:8786
```

## Image Size Optimization

The image uses uv for fast dependency installation and includes:
- Python 3.12 slim base (~200MB)
- UV package manager (~10MB)
- Raw2Zarr dependencies (~500-800MB)

**Total size: ~1-1.2GB** (vs ~2-3GB with conda)

## Troubleshooting

### Permission issues with volumes
```bash
# Run with your user ID
docker run --user $(id -u):$(id -g) \
  -v $(pwd)/data:/opt/raw2zarr/data \
  raw2zarr:latest
```

### JupyterLab not accessible
- Check if port 8888 is already in use: `lsof -i :8888`
- Use a different port: `-p 8889:8888`

### Out of memory errors
```bash
# Increase Docker memory limit
docker run --memory=8g raw2zarr:latest
```

## CI/CD Integration

### GitHub Actions example
```yaml
- name: Build Docker image
  run: docker build -t raw2zarr:${{ github.sha }} .

- name: Run tests in Docker
  run: docker run raw2zarr:${{ github.sha }} uv run pytest
```

### Push to Docker Hub
```bash
# Login
docker login

# Tag
docker tag raw2zarr:latest username/raw2zarr:latest

# Push
docker push username/raw2zarr:latest
```

## References

- [Raw2Zarr Documentation](https://github.com/aladinor/raw2zarr)
- [Radar DataTree Paper](https://doi.org/10.48550/arXiv.2510.24943)
- [UV Documentation](https://docs.astral.sh/uv/)
