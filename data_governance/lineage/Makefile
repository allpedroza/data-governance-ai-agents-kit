# Data Lineage AI Agent - Makefile
# Automation for common development tasks

.PHONY: help install clean test lint format docs run docker analyze

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
VENV := venv
APP := app.py
PORT := 8501

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(BLUE)Data Lineage AI Agent - Development Tasks$(NC)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage: make [target]"

# Setup and Installation
install: ## Install all dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	. $(VENV)/bin/activate && $(PIP) install pytest black flake8 mypy pre-commit
	. $(VENV)/bin/activate && pre-commit install
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

setup: install-dev ## Complete development setup
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	mkdir -p uploads outputs logs examples tests/fixtures
	cp .env.example .env 2>/dev/null || true
	@echo "$(GREEN)✓ Development environment ready$(NC)"

# Cleaning
clean: ## Clean temporary files and caches
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	find . -type d -name '*.egg-info' -delete
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	rm -rf uploads/* outputs/* logs/*
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean ## Deep clean including virtual environment
	@echo "$(YELLOW)Deep cleaning...$(NC)"
	rm -rf $(VENV)
	rm -rf node_modules
	docker-compose down -v 2>/dev/null || true
	@echo "$(GREEN)✓ Deep clean complete$(NC)"

# Testing
test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	. $(VENV)/bin/activate && pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	. $(VENV)/bin/activate && pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(YELLOW)Running tests in watch mode...$(NC)"
	. $(VENV)/bin/activate && pytest-watch tests/

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	. $(VENV)/bin/activate && pytest tests/test_lineage.py::TestIntegration -v

# Code Quality
lint: ## Run linting checks
	@echo "$(YELLOW)Running linters...$(NC)"
	. $(VENV)/bin/activate && flake8 . --max-line-length=100 --exclude=venv,build,dist
	. $(VENV)/bin/activate && mypy . --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	. $(VENV)/bin/activate && black . --line-length=100 --exclude=venv
	. $(VENV)/bin/activate && isort . --skip venv
	@echo "$(GREEN)✓ Code formatted$(NC)"

check-format: ## Check code formatting without changes
	@echo "$(YELLOW)Checking code format...$(NC)"
	. $(VENV)/bin/activate && black . --check --line-length=100 --exclude=venv
	. $(VENV)/bin/activate && isort . --check-only --skip venv

quality: lint test ## Run all quality checks
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

# Documentation
docs: ## Build documentation
	@echo "$(YELLOW)Building documentation...$(NC)"
	cd docs && make clean && make html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html$(NC)"

docs-serve: docs ## Build and serve documentation
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Running the Application
run: ## Run the web interface
	@echo "$(BLUE)Starting Data Lineage Agent at http://localhost:$(PORT)$(NC)"
	. $(VENV)/bin/activate && streamlit run $(APP) --server.port=$(PORT)

run-cli: ## Show CLI help
	@echo "$(BLUE)Data Lineage CLI:$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) main.py --help

run-example: ## Run with example data
	@echo "$(YELLOW)Running with example data...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) main.py init --example ecommerce
	. $(VENV)/bin/activate && $(PYTHON) main.py analyze lineage_example_ecommerce/*.py -o example_output.json
	@echo "$(GREEN)✓ Example analysis complete$(NC)"

# Docker Operations
docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	docker build -t data-lineage-agent:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: docker-build ## Run in Docker container
	@echo "$(BLUE)Running in Docker at http://localhost:$(PORT)$(NC)"
	docker run -p $(PORT):$(PORT) -v $$(pwd)/uploads:/app/uploads data-lineage-agent:latest

docker-compose-up: ## Start all services with docker-compose
	@echo "$(YELLOW)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "Access at: http://localhost:$(PORT)"

docker-compose-down: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs: ## Show Docker logs
	docker-compose logs -f lineage-web

# Analysis Tasks
analyze-pipeline: ## Analyze sample pipeline
	@echo "$(YELLOW)Analyzing sample pipeline...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) main.py analyze examples/ecommerce/*.py examples/ecommerce/*.sql -o analysis.json
	@echo "$(GREEN)✓ Analysis saved to analysis.json$(NC)"

analyze-impact: ## Run impact analysis example
	@echo "$(YELLOW)Running impact analysis...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) main.py impact bronze.orders -p examples/ecommerce
	@echo "$(GREEN)✓ Impact analysis complete$(NC)"

compare-versions: ## Compare two pipeline versions
	@echo "$(YELLOW)Comparing pipeline versions...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) main.py compare examples/v1 examples/v2 -o comparison.json
	@echo "$(GREEN)✓ Comparison saved to comparison.json$(NC)"

# Development Tools
shell: ## Open Python shell with modules loaded
	@echo "$(BLUE)Opening Python shell...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) -i -c "from data_lineage_agent import *; from visualization_engine import *; print('Modules loaded')"

jupyter: ## Start Jupyter lab
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	. $(VENV)/bin/activate && pip install jupyterlab && jupyter lab

profile: ## Profile the application
	@echo "$(YELLOW)Profiling application...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) -m cProfile -o profile.stats main.py analyze examples/ecommerce/*.py
	. $(VENV)/bin/activate && $(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

benchmark: ## Run performance benchmarks
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) tests/benchmark.py
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

# Git Helpers
pre-commit: format lint test ## Run pre-commit checks
	@echo "$(GREEN)✓ Ready to commit$(NC)"

tag-release: ## Create a new release tag
	@echo "$(YELLOW)Creating release tag...$(NC)"
	@read -p "Enter version (e.g., v1.0.0): " version; \
	git tag -a $$version -m "Release $$version"; \
	git push origin $$version
	@echo "$(GREEN)✓ Release tag created$(NC)"

# CI/CD
ci: clean install-dev lint test ## Run CI pipeline locally
	@echo "$(GREEN)✓ CI pipeline passed$(NC)"

deploy-docs: docs ## Deploy documentation to GitHub Pages
	@echo "$(YELLOW)Deploying documentation...$(NC)"
	cd docs/_build/html && git init && git add . && git commit -m "Deploy docs" && \
	git push -f https://github.com/USER/REPO.git master:gh-pages
	@echo "$(GREEN)✓ Documentation deployed$(NC)"

# Utilities
check-deps: ## Check for outdated dependencies
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	. $(VENV)/bin/activate && pip list --outdated

upgrade-deps: ## Upgrade all dependencies
	@echo "$(YELLOW)Upgrading dependencies...$(NC)"
	. $(VENV)/bin/activate && pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U

security-check: ## Run security checks
	@echo "$(YELLOW)Running security checks...$(NC)"
	. $(VENV)/bin/activate && pip install safety bandit
	. $(VENV)/bin/activate && safety check
	. $(VENV)/bin/activate && bandit -r . -ll -x /venv/
	@echo "$(GREEN)✓ Security checks complete$(NC)"

stats: ## Show project statistics
	@echo "$(BLUE)Project Statistics:$(NC)"
	@echo "Lines of code:"
	@find . -name '*.py' -not -path './venv/*' -not -path './.git/*' | xargs wc -l | tail -1
	@echo ""
	@echo "Number of Python files:"
	@find . -name '*.py' -not -path './venv/*' -not -path './.git/*' | wc -l
	@echo ""
	@echo "Number of tests:"
	@grep -r "def test_" tests/ | wc -l

# Quick Commands
quick-start: setup run ## Quick start for new developers

refresh: clean install run ## Clean, reinstall, and run

all: clean setup quality docs ## Run all setup and quality checks

.DEFAULT_GOAL := help
