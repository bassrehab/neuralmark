#!/bin/bash

# Activate virtual environment (if using one)
source venv/bin/activate

# Run basic tests
echo "Running basic tests..."
python main.py --mode test

# Run benchmark
echo "Running benchmark..."
python main.py --mode benchmark

# Run cross-validation
echo "Running cross-validation..."
python main.py --mode cross_validation