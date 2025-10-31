.PHONY: setup run eval test
VENV=venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

run:
	$(PY) src/run_model.py --model $(MODEL)

eval:
	$(PY) src/evaluate.py

test:
	$(PY) -m pytest -q || true
