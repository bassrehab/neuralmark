Usage instructions:

## Initial Setup

- **Remove old environment if exists**
```rm -rf venv```

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

**Upgrade pip**
```pip install --upgrade pip```

**Clear pip cache** (optional, but can help avoid version conflicts)
```pip cache purge```




## Installation notes for different platforms:

1. **macOS (M1/M2)**:
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

2. **Linux with NVIDIA GPU**:
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

3. **Windows**:
```bash
# Make sure Python 3.11 is installed from python.org

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements-windows-cpu.txt  # or requirements-windows-gpu.txt
```

## Setup .env

create ```.env``` file with the following values.

```
UNSPLASH_ACCESS_KEY=your-unsplash-key
PRIVATE_KEY=your-secure-private-key-here
```

## Basic test run
```python main.py --mode test```

**Or for benchmark**
```python main.py --mode benchmark```

**Or for cross-validation**
```python main.py --mode cross_validation```

**You can specify custom config file**
```python main.py --mode test --config custom_config.yaml```

**You can specify custom output directory**
```python main.py --mode test --output custom_output```

**Or use the shell script (make it executable first):**
```bash
chmod +x run_tests.sh
./run_tests.sh 
```


## Cleanups (optional)

**Pre-run cleanup**
```python cleanup.py --pre-run```

**Post-run cleanup**
```python cleanup.py --post-run```