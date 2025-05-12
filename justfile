# List available commands
default:
    @just --list

# Install dependencies
install:
    poetry install

# Run the service locally
run:
    poetry run uvicorn src.api.main:app --reload

# Run tests
test:
    poetry run pytest

# Run tests with coverage
test-cov:
    poetry run pytest --cov=src tests/

# Format code
format:
    poetry run black src tests
    poetry run isort src tests

# Type check
type-check:
    poetry run mypy src

# Build Docker image for linux/amd64
docker-build:
    docker build --platform linux/amd64 -t tbished/finance-tracker-ml:latest .

# Push Docker image to registry
docker-push:
    docker push tbished/finance-tracker-ml:latest

# Build and push Docker image
docker-build-push: docker-build docker-push

# Run Docker container
docker-run:
    docker run -p 8000:8000 tbished/finance-tracker-ml:latest

# Clean up Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -r {} +
    find . -type d -name "*.egg" -exec rm -r {} +
    find . -type d -name ".pytest_cache" -exec rm -r {} +
    find . -type d -name ".mypy_cache" -exec rm -r {} +
    find . -type d -name ".coverage" -exec rm -r {} +
    find . -type d -name "htmlcov" -exec rm -r {} +
    find . -type d -name "dist" -exec rm -r {} +
    find . -type d -name "build" -exec rm -r {} + 