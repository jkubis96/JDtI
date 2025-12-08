.PHONY: format lint all

format:
	isort jdti
	black jdti
	isort tests
	black tests


lint:
	pylint --exit-zero jdti/jdti.py
	pylint --exit-zero jdti/utils.py
	pylint --exit-zero tests/test_utils.py
	pylint --exit-zero tests/test_subclustering.py
	pylint --exit-zero tests/test_integration.py
	pylint --exit-zero tests/test_clustering.py




all: format lint
