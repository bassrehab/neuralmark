# AlphaPunch - Robust Image Fingerprinting System

## Overview
AlphaPunch is an advanced image fingerprinting system designed to verify image ownership and detect modifications. It embeds invisible digital fingerprints into images and can later verify these fingerprints to:
- Authenticate image ownership
- Detect unauthorized modifications
- Track image manipulations (e.g., compression, rotation, cropping)
- Adapt to image content using neural attention

## Key Features
- Multi-domain fingerprinting (spatial, frequency, and wavelet domains)
- Neural attention-based adaptive embedding
- Robust against common image manipulations
- Modification detection and classification
- Comprehensive testing and validation framework
- Performance monitoring and visualization

## System Architecture

### Core Components

1. **Algorithm (`algorithm.py`)**
   - Main interface for fingerprinting operations
   - Neural attention integration
   - Handles high-level fingerprint generation and verification
   - Coordinates between different processing domains

2. **Neural Attention (`neural_attention.py`)**
   - Attention-based feature extraction
   - Adaptive embedding strength
   - Content-aware fingerprinting
   - Multi-scale attention processing

3. **Adaptive Multi-Domain Fingerprinting (`amdf.py`)**
   - Implements core fingerprinting algorithms
   - Manages multi-domain feature extraction
   - Handles wavelet transformations and frequency analysis
   - Provides robust fingerprint comparison methods

4. **Fingerprint Core (`fingerprint_core.py`)**
   - Low-level fingerprinting operations
   - DCT and wavelet transformations
   - Basic feature extraction and comparison

5. **Image Author (`author.py`)**
   - Manages image ownership and verification
   - Maintains fingerprint database
   - Handles fingerprint storage and retrieval

### Testing and Utilities

1. **Test Framework**
   - Base testing functionality (`test_base.py`)
   - Core fingerprint tests (`test_core.py`)
   - Neural attention tests (`test_neural_attention.py`)
   - AMDF testing (`test_amdf.py`)
   - Integration testing (`test_integration.py`)
   - Performance benchmarking
   - Cross-validation testing

2. **Utilities (`utils.py`)**
   - Configuration management
   - Logging setup
   - Test image acquisition
   - Image manipulation utilities
   - Attention visualization tools

### Directory Structure
```
alphapunch/
├── config.yaml              # Configuration settings
├── main.py                 # Main entry point
├── Makefile               # Build and test automation
├── setup.py               # Installation setup
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables (private)
├── .env.example         # Environment template
├── alphapunch/           # Core package
│   ├── __init__.py
│   ├── author.py
│   ├── algorithm.py
│   └── core/
│       ├── __init__.py
│       ├── amdf.py
│       ├── neural_attention.py
│       └── fingerprint_core.py
├── tests/               # Test suite
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_core.py
│   ├── test_neural_attention.py
│   ├── test_amdf.py
│   └── test_integration.py
├── utils.py              # Utility functions
└── output/              # Generated outputs
    ├── reports/        # Test reports
    ├── plots/         # Visualizations
    └── fingerprints/  # Generated fingerprints
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alphapunch.git
cd neuralmark
```

2. Install dependencies:
```bash
make install
```

3. Create environment file:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Usage

### Using Make Commands

```bash
# Install dependencies
make install

# Run tests
make test

# Clean generated files
make clean

# Run in different modes
make run        # Normal test mode
make benchmark  # Performance benchmark mode
make crossval   # Cross-validation mode
```

### Using Python Directly

```python
# Basic usage
python main.py --mode test

# Benchmark performance
python main.py --mode benchmark

# Cross-validation
python main.py --mode cross_validation
```

## Neural Attention Features

The system now includes neural attention-based improvements:

1. **Adaptive Embedding**
   - Content-aware fingerprint embedding
   - Multiple attention layers
   - Automatic strength adjustment

2. **Enhanced Robustness**
   - Multi-scale feature extraction
   - Attention-weighted verification
   - Improved modification detection

3. **Visual Analysis**
   - Attention map visualization
   - Feature importance analysis
   - Modification localization

## Key Technologies
- TensorFlow for neural attention and deep feature extraction
- OpenCV for image processing
- PyWavelets for wavelet transformations
- NumPy for numerical operations
- Matplotlib for visualization

## Output and Reports
- Detailed test reports in JSON format
- Performance visualizations
- Modification detection reports
- Success rate analysis by scenario
- Attention map visualizations
- Feature importance plots

The system is designed to be modular, extensible, and configurable through the `config.yaml` file, making it adaptable to different use cases and requirements.

## Environment Variables
Required environment variables in `.env`:
```
UNSPLASH_ACCESS_KEY=your_unsplash_key
PRIVATE_KEY=your_secure_private_key
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Run tests: `make test`
4. Commit changes
5. Push to branch
6. Create Pull Request

----

See [USAGE.md](./USAGE.md) for detailed configuration and usage information.

**Author:** Subhadip Mitra