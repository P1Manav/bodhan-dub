.PHONY: help setup check lint test smoke estimate clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make setup     Install dependencies in editable mode"
	@echo "  make check     Run linters and test suite"
	@echo "  make lint      Run ruff check and formatting"
	@echo "  make test      Run pytest unit and offline tests"
	@echo "  make smoke     Run end-to-end smoke test on synthetic fixture"
	@echo "  make estimate  Estimate cost for a sample input"
	@echo "  make clean     Remove cache and temporary build files"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

check: lint test

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest tests -v

smoke:
	$(PYTHON) -m pytest tests/e2e/test_smoke.py -v

estimate:
	$(PYTHON) -m bodhan_dub --input tests/fixtures/sample.mp4 --langs hi,ta,bn --estimate-only

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	rm -rf build dist *.egg-info
