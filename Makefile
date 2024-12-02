.PHONY: install test clean run run-amdf run-cdha run-comparison debug benchmark crossval

install:
	python setup.py

test:
	python -m pytest tests

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.pyc" -delete
	rm -rf output/* database/* downloads/* plots/* reports/* cache/* tmp/* logs/*

# Run commands for different algorithms
run:
	python main.py --mode test

run-amdf:
	python main.py --mode test --algorithm amdf

run-cdha:
	python main.py --mode test --algorithm cdha

run-comparison:
	python main.py --mode test --algorithm both

debug:
	python main.py --mode test --log-level DEBUG

benchmark:
	python main.py --mode benchmark

crossval:
	python main.py --mode cross_validation
