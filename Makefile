.PHONY: format lint check all

format:
	isort scdiff
	black scdiff
	isort tests
	black tests


lint:
	pylint --exit-zero scdiff/scdiff.py
	pylint --exit-zero scdiff/utils.py
	pylint --exit-zero tests/test_utils.py
	pylint --exit-zero tests/test_subclustering.py
	pylint --exit-zero tests/test_integration.py
	pylint --exit-zero tests/test_clustering.py




all: format lint
