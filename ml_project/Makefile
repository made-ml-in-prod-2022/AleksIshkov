PYTHON?=python

all: help

help:
	@echo "report - create report"

report:
	@echo "creating report and store in ./notebooks/heart_disease_cleveland/report/"
	$(PYTHON) notebooks/heart_disease_cleveland/report/Report.py

train:
	poetry run cli train

test:
	poetry run pytest ./tests/unit/* -vv


