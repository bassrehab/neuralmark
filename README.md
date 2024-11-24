# AlphaPunch - Robust Image Fingerprinting System

## Overview
AlphaPunch is an advanced image fingerprinting system designed to verify image ownership and detect modifications. It embeds invisible digital fingerprints into images and can later verify these fingerprints to:
- Authenticate image ownership
- Detect unauthorized modifications
- Track image manipulations (e.g., compression, rotation, cropping)

## Key Features
- Multi-domain fingerprinting (spatial, frequency, and wavelet domains)
- Robust against common image manipulations
- Modification detection and classification
- Comprehensive testing and validation framework
- Performance monitoring and visualization

## System Architecture

### Core Components

1. **Enhanced Algorithm (`enhanced_algorithm.py`)**
   - Main interface for fingerprinting operations
   - Handles high-level fingerprint generation and verification
   - Coordinates between different processing domains

2. **Adaptive Multi-Domain Fingerprinting (`amdf.py`)**
   - Implements core fingerprinting algorithms
   - Manages multi-domain feature extraction
   - Handles wavelet transformations and frequency analysis
   - Provides robust fingerprint comparison methods

3. **Fingerprint Core (`fingerprint_core.py`)**
   - Low-level fingerprinting operations
   - DCT and wavelet transformations
   - Basic feature extraction and comparison

4. **Image Author (`author.py`)**
   - Manages image ownership and verification
   - Maintains fingerprint database
   - Handles fingerprint storage and retrieval

### Testing and Utilities

1. **Test Framework (`test_authorship.py`)**
   - Comprehensive testing suite
   - Performance benchmarking
   - Cross-validation testing
   - Report generation and visualization

2. **Utilities (`utils.py`)**
   - Configuration management
   - Logging setup
   - Test image acquisition
   - Image manipulation utilities

### Directory Structure
```
alphapunch/
├── config.yaml              # Configuration settings
├── main.py                 # Main entry point
├── setup.py               # Installation setup
├── requirements/          # Platform-specific requirements
├── alphapunch/           # Core package
│   ├── __init__.py
│   ├── author.py
│   ├── enhanced_algorithm.py
│   └── core/
│       ├── __init__.py
│       ├── amdf.py
│       └── fingerprint_core.py
├── utils.py              # Utility functions
├── test_authorship.py    # Testing framework
└── output/              # Generated outputs
    ├── reports/        # Test reports
    ├── plots/         # Visualizations
    └── fingerprints/  # Generated fingerprints
```

## How It Works

1. **Fingerprint Generation**
   - Extract features from multiple image domains
   - Generate unique fingerprint using private key
   - Embed fingerprint invisibly into image

2. **Verification Process**
   - Extract fingerprint from suspect image
   - Compare with stored fingerprints
   - Analyze similarities and detect modifications
   - Generate verification report

3. **Testing and Validation**
   - Test against various image manipulations
   - Cross-validate fingerprint robustness
   - Generate performance metrics
   - Visualize results

## Usage

```python
# Basic usage
python main.py --mode test

# Benchmark performance
python main.py --mode benchmark

# Cross-validation
python main.py --mode cross_validation
```

## Key Technologies
- TensorFlow for deep feature extraction
- OpenCV for image processing
- PyWavelets for wavelet transformations
- NumPy for numerical operations
- Matplotlib for visualization

## Output and Reports
- Detailed test reports in JSON format
- Performance visualizations
- Modification detection reports
- Success rate analysis by scenario

The system is designed to be modular, extensible, and configurable through the `config.yaml` file, making it adaptable to different use cases and requirements.

----

See [USAGE.md](./USAGE.md) for various configurations and usage details

**Author:** Subhadip Mitra