PY = python3

all: download build features train evaluate

download:
	$(PY) -m src.download

build:
	$(PY) -m src.build_dataset

features:
	$(PY) -m src.features

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

test:
	$(PY) -m pytest tests/ -q

.PHONY: all download build features train evaluate test