# InferScale Makefile

.PHONY: help install test lint typecheck format dev train-smoke evaluate benchmark clean docker-build docker-up docker-down docker-logs

# Default target
help:
	@echo "InferScale - Autonomous LLM Inference Autoscaling with RL"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install Python dependencies"
	@echo "  test          - Run all tests"
	@echo "  lint          - Run linting (ruff)"
	@echo "  typecheck     - Run type checking (mypy)"
	@echo "  format        - Format code (black)"
	@echo "  dev           - Start development servers (API + frontend)"
	@echo "  train-smoke   - Run short PPO training smoke test"
	@echo "  evaluate      - Run policy evaluation benchmark"
	@echo "  benchmark     - Run full benchmark suite"
	@echo "  clean         - Clean build artifacts"
	@echo "  docker-build  - Build Docker images"
	@echo "  docker-up     - Start Docker Compose stack"
	@echo "  docker-down   - Stop Docker Compose stack"
	@echo "  docker-logs   - View Docker Compose logs"

# Python environment
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,test]"
	@touch $(VENV)/bin/activate

test: install
	$(PYTHON) -m pytest backend/inferscale/tests -v --cov=inferscale --cov-report=term-missing

lint:
	$(VENV)/bin/ruff check backend/inferscale

typecheck:
	$(VENV)/bin/mypy backend/inferscale

format:
	$(VENV)/bin/black backend/inferscale

# Development servers
dev: install
	@echo "Starting API server on http://localhost:8000"
	@echo "Starting frontend on http://localhost:5173"
	@$(MAKE) -j2 dev-api dev-frontend

dev-api: install
	cd backend && $(PYTHON) -m uvicorn inferscale.api.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	cd frontend && npm run dev

# Training and evaluation
train-smoke: install
	cd backend && $(PYTHON) -m inferscale.cli train --timesteps 5000 --short

evaluate: install
	cd backend && $(PYTHON) -m inferscale.cli evaluate --episodes 10

benchmark: install
	cd backend && $(PYTHON) -m inferscale.cli benchmark --output results/benchmark_$(date +%Y%m%d_%H%M%S).json

# Docker
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Cleanup
clean:
	rm -rf $(VENV)
	rm -rf backend/__pycache__ backend/inferscale/__pycache__ backend/inferscale/**/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info dist build
	rm -rf frontend/node_modules frontend/dist

# Install frontend dependencies
frontend-install:
	cd frontend && npm install

# Generate synthetic dataset
generate-data: install
	cd backend && $(PYTHON) -m inferscale.cli generate-data --output data/synthetic.csv

# Run API tests
test-api: install
	$(PYTHON) -m pytest backend/inferscale/tests/test_api.py -v

# Run simulator tests
test-simulator: install
	$(PYTHON) -m pytest backend/inferscale/tests/test_simulator.py -v

# Run policy tests
test-policies: install
	$(PYTHON) -m pytest backend/inferscale/tests/test_policies.py -v

# Run RL tests
test-rl: install
	$(PYTHON) -m pytest backend/inferscale/tests/test_rl.py -v