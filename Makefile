.PHONY: install test clean run benchmark

install:
	pip install -r requirements.txt

test:
	python -m unittest discover tests

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.pyc" -delete
	rm -rf output/* database/* downloads/* plots/* reports/* cache/* tmp/* logs/*

run:
	python main.py --mode test

benchmark:
	python main.py --mode benchmark

crossval:
	python main.py --mode cross_validation