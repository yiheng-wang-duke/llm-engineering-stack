.PHONY: install run demo test lint smoke
install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.lock
	.venv/bin/python -m pip install -e '.[dev]'
run:
	.venv/bin/python -m workbench
demo:
	AW_MODE=demo .venv/bin/python -m workbench
test:
	.venv/bin/python -m pytest -q
lint:
	.venv/bin/ruff check --config pyproject.toml workbench tests scripts
smoke:
	.venv/bin/python scripts/smoke.py
