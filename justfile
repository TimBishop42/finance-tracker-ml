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

# Build Docker image
docker-build:
    docker build -t finance-tracker-ml .

# Run Docker container
docker-run:
    docker run -p 8000:8000 finance-tracker-ml

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