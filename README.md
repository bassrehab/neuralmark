# NeuralMark - Robust Image Fingerprinting System

## Overview
NeuralMark is an advanced image fingerprinting system that verifies image ownership and detects modifications through invisible digital fingerprints. The system:
- Authenticates image ownership
- Detects unauthorized modifications
- Tracks image manipulations (compression, rotation, cropping)
- Uses neural attention for content-adaptive fingerprinting

## Key Features
- Multi-domain fingerprinting (spatial, frequency, wavelet)
- Neural attention-based adaptive embedding
- Robust against common image manipulations
- Modification detection and classification
- Comprehensive testing framework
- Performance monitoring and visualization

## System Architecture

### Core Components (`core/`)
- **AMDF** (`amdf.py`): Core fingerprinting algorithms, multi-domain feature extraction
- **Neural Attention** (`neural_attention.py`): Attention-based feature extraction and adaptive embedding
- **Fingerprint Core** (`fingerprint_core.py`): Low-level fingerprinting operations
- **Cache Utils** (`cache_utils.py`): Caching functionality for performance optimization

### Utilities (`utils/`)
- **Config** (`config.py`): Configuration management
- **Image** (`image.py`): Image processing utilities
- **Logging** (`logging.py`): Logging setup and management
- **Visualization** (`visualization.py`): Result visualization tools

### Base Files
- **Algorithm** (`algorithm.py`): Main fingerprinting interface
- **Author** (`author.py`): Image ownership management
- **Main** (`main.py`): Program entry point
- **Config** (`config.yaml`): System configuration
- **Setup** (`setup.py`): Installation configuration
- **Cleanup** (`cleanup.py`): Directory maintenance
- **TF Config** (`tf_config.py`): TensorFlow configuration

## Directory Structure
```
neuralmark/
├── core/
│   ├── __init__.py
│   ├── amdf.py
│   ├── cache_utils.py
│   ├── fingerprint_core.py
│   └── neural_attention.py
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── image.py
│   ├── logging.py
│   └── visualization.py
│
├── cleanup.py
├── config.yaml
├── main.py
├── Makefile
├── setup.py
├── test_authorship.py
└── tf_config.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neuralmark.git
cd neuralmark
```

2. Install dependencies:
```bash
make install
```

## Usage

### Using Make Commands
```bash
make install    # Install dependencies
make test      # Run tests
make clean     # Clean generated files
make run       # Test mode
make debug     # Debug mode
make benchmark # Performance benchmark
make crossval  # Cross-validation
```

### Using Python Directly
```bash
python main.py --mode test
python main.py --mode benchmark
python main.py --mode cross_validation
python main.py --log-level DEBUG
```

## Configuration
The system is configured through `config.yaml`, which includes:
- Directory paths
- Algorithm parameters
- Testing configuration
- Resource allocation
- Logging settings

## Key Technologies
- TensorFlow
- OpenCV
- PyWavelets
- NumPy
- Matplotlib

## Output
The system generates:
- Test reports (JSON)
- Performance visualizations
- Success rate analysis
- Attention map visualizations
- Feature importance plots

## Environment Variables
Required in `.env`:
```
UNSPLASH_ACCESS_KEY=your_unsplash_key
PRIVATE_KEY=your_secure_private_key
```