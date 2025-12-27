SHELL := /bin/bash
PATH := $(HOME)/.local/bin:$(PATH)

install:
	@if ! command -v uv &> /dev/null; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv sync

test:
	uv run python -m pytest tests/ -vv --cov=logic --cov=api

format:	
	uv run black logic/*.py api/*.py #*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*?\.py logic/*.py api/*.py 

refactor: format lint

all: install format lint test
