# NeuralMark

NeuralMark is an advanced image fingerprinting system that uses neural attention and multi-domain processing to embed and verify robust digital fingerprints in images.

## Features
Refer to [ALGORITHM.md](./algorithm.md) for more

- **Multiple Fingerprinting Algorithms:**
  - AMDF (Adaptive Multi-Domain Fingerprinting)
  - CDHA (Cross-Domain Hierarchical Attention)
  - Comparison mode for algorithm evaluation

- **Advanced Neural Attention:**
  - Cross-domain feature integration
  - Hierarchical feature fusion
  - Adaptive weight generation
  - Content-aware processing

- **Robust Protection:**
  - Multi-scale feature extraction
  - Resistance to common image manipulations
  - Attack detection and classification
  - Modification tracking

- **Comprehensive Analysis:**
  - Detailed verification reports
  - Visualization tools
  - Performance metrics
  - Cross-validation capabilities

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/neuralmark.git
cd neuralmark
```


### 2. Virtual Env setup

#### Remove old environment if exists
```rm -rf venv```


#### Create virtual environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows

# Install requirements
python setup.py  # Basic installation
python setup.py --dev  # Include development tools
python setup.py --docs  # Include documentation tools

# Or install directly using the appropriate requirements file
pip install -r requirements-macos-arm.txt  # For M1/M2 Macs
pip install -r requirements-linux-gpu.txt  # For Linux with GPU
# etc.
```

#### Upgrade pip

```pip install --upgrade pip```
or ```pip3 install --upgrade pip```

#### Clear pip cache 
(optional, but can help avoid version conflicts)

```pip cache purge```

#### Installation notes for different platforms

- **macOS (M1/M2)**:
  ```bash
  # Install Homebrew if not already installed
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Install Python 3.11
  brew install python@3.11
  
  # Create and activate virtual environment
  python3.11 -m venv venv
  source venv/bin/activate
  
  # Install requirements
  pip install -r requirements-macos-arm.txt
  ```

- **Linux with NVIDIA GPU**:
  ```bash
  # Install CUDA dependencies first
  sudo apt update
  sudo apt install nvidia-cuda-toolkit nvidia-cudnn
  
  # Create and activate virtual environment
  python3 -m venv venv
  source venv/bin/activate
  
  # Install requirements
  pip install -r requirements-linux-gpu.txt
  ```

- **Windows**:
  ```bash
  # Make sure Python 3.11 is installed from python.org
  
  # Create and activate virtual environment
  python -m venv venv
  .\venv\Scripts\activate
  
  # Install requirements
  pip install -r requirements-windows-cpu.txt  # or requirements-windows-gpu.txt
  ```


### 3. Set up environment variables:
  ```bash
  cp .env.example .env
  # Edit .env with your credentials
  ```

### 4. Install dependencies:
```bash
make install
```

## Usage

### Basic Usage

```python
from neuralmark.algorithm import create_neural_mark
from neuralmark.utils import load_config, setup_logger

# Load configuration
config = load_config('config.yaml')
logger = setup_logger(config)

# Create NeuralMark instance (AMDF or CDHA)
neural_mark = create_neural_mark('cdha', private_key, logger, config)

# Generate fingerprint
fingerprint = neural_mark.generate_fingerprint(image)

# Embed fingerprint
fingerprinted_image = neural_mark.embed_fingerprint(image, fingerprint)

# Verify fingerprint
is_authentic, similarity, modifications = neural_mark.verify_fingerprint(
    suspect_image, fingerprint
)
```

### Command Line Interface

I'll show you the updated Makefile that includes the run command and related commands:

```makefile
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
```

Let me update the README.md section about the run commands:



### Command Line Interface

```bash
# Basic run (uses algorithm from config.yaml)
make run

# Run with specific algorithm
make run-amdf    # Run AMDF algorithm
make run-cdha    # Run CDHA algorithm
make run-comparison  # Run both algorithms for comparison

# Debug mode
make debug

# Run benchmarks
make benchmark

# Run cross-validation
make crossval

# Clean generated files
make clean
```



### Advanced Usage

```bash
# Run with specific algorithm
python main.py --algorithm cdha --mode test

# Run comparison mode
python main.py --algorithm both --mode test

# Debug mode
python main.py --mode test --log-level DEBUG
```

## Project Structure

```
neuralmark/
├── core/
│   ├── amdf.py              # AMDF implementation
│   ├── cache_utils.py       # Caching functionality
│   ├── fingerprint_core.py  # Core fingerprinting operations
│   └── neural_attention.py  # Neural attention mechanisms
│
├── utils/
│   ├── config.py           # Configuration management
│   ├── image.py           # Image processing utilities
│   ├── logging.py         # Logging setup
│   └── visualization.py   # Visualization tools
│
├── algorithm.py           # Main algorithm interface
├── cleanup.py            # Directory maintenance
├── config.yaml           # Configuration file
├── main.py              # Program entry point
├── Makefile             # Build automation
├── setup.py             # Installation setup
├── test_authorship.py   # Authorship testing
└── tf_config.py         # TensorFlow configuration
```

## Configuration

The system is configured through `config.yaml`, which includes:

- Algorithm selection and parameters
- Neural network architecture settings
- Testing configuration
- Resource allocation
- Logging preferences

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
python -m pytest

# Run specific test category
python -m pytest tests/test_core
python -m pytest tests/test_utils

# Run with coverage
python -m pytest --cov=neuralmark
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.6+
- NumPy 1.22+
- Additional requirements in requirements/*.txt

## Platform Support

- Linux (CPU/GPU)
- Windows (CPU/GPU)
- macOS (Intel/ARM)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests: `make test`
4. Commit changes
5. Push to branch
6. Create Pull Request

## License
TODO

## Authors
Subhadip Mitra