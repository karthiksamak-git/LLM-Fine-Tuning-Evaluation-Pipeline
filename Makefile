PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PROFILE ?= cpu

.PHONY: help setup test lint prepare baseline train eval ablate efficiency analyze export ingest web-dev web-build all clean

help:
	@echo "Targets:"
	@echo "  setup      install python (venv) and web dependencies"
	@echo "  test       run the pytest suite (CPU unit tests)"
	@echo "  lint       ruff check"
	@echo "  prepare    build splits + data card"
	@echo "  baseline   evaluate the base model (zero-shot + few-shot)"
	@echo "  train      LoRA fine-tune (+ merge + parity check)"
	@echo "  eval       evaluate fine-tuned adapter and merged model"
	@echo "  ablate     run the ablation grids"
	@echo "  efficiency latency/size/precision measurements"
	@echo "  analyze    failure taxonomy + curated examples"
	@echo "  export     write web JSON, figures and README tables"
	@echo "  ingest     install a results zip from a GPU notebook (ZIP=path/to.zip)"
	@echo "  web-dev    run the Next.js dashboard"
	@echo "  web-build  build the Next.js dashboard"
	@echo "  all        full pipeline for PROFILE (default cpu)"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	cd web && npm ci

test:
	$(PY) -m pytest tests -o addopts="" -q

lint:
	.venv/bin/ruff check .

prepare:
	$(PY) -m mlfte.cli prepare --profile $(PROFILE)

baseline:
	$(PY) -m mlfte.cli baseline --profile $(PROFILE)

train:
	$(PY) -m mlfte.cli train --profile $(PROFILE)

eval:
	$(PY) -m mlfte.cli eval --profile $(PROFILE)

ablate:
	$(PY) -m mlfte.cli ablate --profile $(PROFILE)

efficiency:
	$(PY) -m mlfte.cli efficiency --profile $(PROFILE)

analyze:
	$(PY) -m mlfte.cli analyze --profile $(PROFILE)

export:
	$(PY) -m mlfte.cli export --profile $(PROFILE)

ingest:
	@test -n "$(ZIP)" || (echo "usage: make ingest ZIP=results.zip PROFILE=gpu" && exit 1)
	$(PY) -m mlfte.cli ingest $(ZIP) --expect-profile $(PROFILE)

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

all:
	$(PY) -m mlfte.cli all --profile $(PROFILE)

clean:
	rm -rf results outputs figures/*.png
