# justfile
test-all: test-py311 test-py312 test-py313 test-py314
    @echo "✅ All Python versions passed!"

test-py311:
    uv sync --python=3.11 --extra dev
    uv run --python=3.11 pytest tests -v

test-py312:
    uv sync --python=3.12 --extra dev
    uv run --python=3.12 pytest tests -v

test-py313:
    uv sync --python=3.13 --extra dev
    uv run --python=3.13 pytest tests -v

test-py314:
    uv sync --python=3.14 --extra dev
    uv run --python=3.14 pytest tests -v
