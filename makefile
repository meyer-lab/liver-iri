
flist = $(wildcard liver-iri/figures/figure*.py)

all: $(patsubst liver-iri/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: liver-iri/figures/figure%.py
	mkdir -p output
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run fbuild $*

test:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run pytest -s -v -x

coverage.xml:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run pytest --cov=liver-iri --cov-report=xml --cov-config=.github/workflows/coveragerc

testprofile:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	rm -rf output
