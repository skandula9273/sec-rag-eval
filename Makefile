.PHONY: help install lock db-init data ingest query eval demo test lint fmt check-numbers

CONFIG ?= configs/v0.yaml

help:
	@echo "install        install package + dev extras (editable)"
	@echo "lock           freeze exact versions to requirements.lock (reproducibility)"
	@echo "db-init        apply db/schema.sql to \$$DATABASE_URL"
	@echo "data           download FinanceBench PDFs into data/ (see README)"
	@echo "ingest         parse -> chunk -> embed -> load to pgvector   (CONFIG=$(CONFIG))"
	@echo "eval           run FinanceBench eval, write timestamped JSON to eval_results/"
	@echo "demo           launch Streamlit demo"
	@echo "test           run pytest"
	@echo "check-numbers  verify every README number matches its committed artifact"
	@echo "lint           ruff check"

install:
	python -m pip install -e ".[dev,demo]"

lock:
	python -m pip freeze > requirements.lock

db-init:
	psql "$(DATABASE_URL)" -f src/sec_rag/db/schema.sql

data:
	python -m sec_rag.ingest.financebench --download --out data/

ingest:
	python -m sec_rag.ingest.load --config $(CONFIG)

eval:
	python -m sec_rag.eval.run_financebench --config $(CONFIG)

demo:
	streamlit run demo/streamlit_app.py

test:
	pytest

# Fail if any README number drifted from the eval_results/*.json that produced it,
# or if a new claim-shaped number was added without registering it. No secrets.
check-numbers:
	python -m sec_rag.eval.check_numbers --strict

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
