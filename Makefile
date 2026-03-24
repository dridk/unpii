.PHONY: build test

build:
	uv run maturin develop

test: build
	uv run pytest tests/python/ -v
