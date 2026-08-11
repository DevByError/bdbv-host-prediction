# =============================================================================
# Makefile for BDBV Host Prediction
# =============================================================================

.PHONY: help install install-dev test lint format clean run-train run-predict run-validate run-figures docker-build docker-up docker-down

# Colors
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

help: ## Show this help message
	@echo '${GREEN}Usage:${RESET}'
	@echo '  make ${YELLOW}<target>${RESET}'
	@echo ''
	@echo '${GREEN}Targets:${RESET}'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-20s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html

test-quick: ## Run tests without coverage
	pytest tests/ -v

test-integration: ## Run integration tests
	pytest tests/test_integration/ -v

lint: ## Run linters
	black --check src/ scripts/ tests/
	flake8 src/ scripts/ tests/
	mypy src/ scripts/ tests/
	pylint src/ scripts/ tests/

format: ## Format code
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf logs/*.log
	rm -rf results/models/*

run-train: ## Run training pipeline
	python scripts/run_training.py

run-predict: ## Run prediction pipeline
	python scripts/run_prediction.py --input data/new_hosts.csv --output results/predictions.csv

run-validate: ## Run validation
	python scripts/run_validation.py --predictions results/predictions.csv

run-figures: ## Generate figures
	python scripts/generate_figures.py

notebook: ## Start Jupyter notebook
	jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

docker-build: ## Build Docker image
	docker build -t bdbv-prediction .

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

api: ## Start API server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

coverage: ## Generate coverage report
	pytest tests/ -v --cov=src --cov-report=html
	open htmlcov/index.html

benchmark: ## Run benchmarks
	pytest tests/test_benchmarks/ -v --benchmark-only