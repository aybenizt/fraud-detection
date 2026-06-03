.PHONY: setup data train serve dashboard test docker-build docker-run clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -m src.data

train:
	$(PYTHON) -m src.pipeline

serve:
	$(PYTHON) -m src.api

dashboard:
	$(VENV)/bin/streamlit run src/app.py

test:
	$(PYTHON) -m pytest tests/ -v

docker-build:
	docker-compose build

docker-run:
	docker-compose up

clean:
	rm -rf data/*.csv models/*.pkl logs/*.log mlruns/ .pytest_cache/
