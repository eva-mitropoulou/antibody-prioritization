PYTHON ?= python

.PHONY: reproduce-small test report

reproduce-small:
	RUN_TESTS=0 PYTHON=$(PYTHON) bash scripts/reproduce_final_reports.sh

test:
	$(PYTHON) -m pytest -q

report:
	PYTHON=$(PYTHON) bash scripts/reproduce_final_reports.sh
