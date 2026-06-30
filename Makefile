PYTHON ?= python

.PHONY: reproduce-small report

reproduce-small:
	PYTHON=$(PYTHON) bash scripts/reproduce_final_reports.sh

report:
	PYTHON=$(PYTHON) bash scripts/reproduce_final_reports.sh
