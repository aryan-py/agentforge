.PHONY: install dev test demo lint format clean

install:
	uv sync

dev:
	uvicorn agentforge.api.app:app --reload --port 8000

test:
	uv run pytest tests/unit -v

test-integration:
	uv run pytest tests/integration -v

test-all:
	uv run pytest tests/ -v

demo:
	uv run python scripts/demo_phase1.py

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

clean:
	rm -rf .cache/ __pycache__/ src/**/__pycache__/ tests/**/__pycache__/ .pytest_cache/

dashboard:
	uv run streamlit run dashboard/app.py
