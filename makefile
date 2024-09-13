SHELL := /bin/bash

.PHONY: clean test

flist = $(filter-out liver_iri/figures/figure2.py liver_iri/figures/figure4.py, $(wildcard liver_iri/figures/figure*.py))
allOutput = $(patsubst liver_iri/figures/figure%.py, output/figure%.svg, $(flist))

.venv:
	rye sync

all: $(allOutput)

output/figure%.svg: .venv liver_iri/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

test: .venv
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 rye run pytest -s -v -x

coverage.xml: .venv
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 rye run pytest --cov=liver-iri --cov-report=xml --cov-config=.github/workflows/coveragerc

testprofile: .venv
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 rye run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	rm -rf output
