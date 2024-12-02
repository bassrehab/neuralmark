import platform
import subprocess
import sys
from pathlib import Path


def get_platform_requirements():
    """Determine the appropriate requirements file based on platform and capabilities."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    requirements_dir = Path('./requirements')
    requirements_dir.mkdir(exist_ok=True)

    # Base requirements always needed
    base_reqs = requirements_dir / 'requirements-base.txt'

    # Platform-specific requirements
    if system == "darwin":  # macOS
        if "arm" in machine:
            platform_reqs = requirements_dir / 'requirements-macos-arm.txt'
        else:
            platform_reqs = requirements_dir / 'requirements-macos-intel.txt'

    elif system == "linux":
        # Check for NVIDIA GPU
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            platform_reqs = requirements_dir / 'requirements-linux-gpu.txt'
        except (subprocess.CalledProcessError, FileNotFoundError):
            platform_reqs = requirements_dir / 'requirements-linux-cpu.txt'

    elif system == "windows":
        # Check for NVIDIA GPU
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            platform_reqs = requirements_dir / 'requirements-windows-gpu.txt'
        except (subprocess.CalledProcessError, FileNotFoundError):
            platform_reqs = requirements_dir / 'requirements-windows-cpu.txt'

    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)

    return base_reqs, platform_reqs


def create_base_requirements():
    """Create base requirements file."""
    base_reqs = [
        "numpy>=1.22.0",
        "opencv-python>=4.6.0",
        "tensorflow>=2.10.0",
        "matplotlib>=3.5.0",
        "PyWavelets>=1.3.0",
        "scikit-image>=0.19.0",
        "tqdm>=4.64.0",
        "python-dotenv>=0.20.0",
        "pyyaml>=6.0",
        "requests>=2.28.0"
    ]

    with open('./requirements/requirements-base.txt', 'w') as f:
        f.write('\n'.join(base_reqs))


def create_platform_requirements():
    """Create platform-specific requirements files."""
    requirements_dir = Path('./requirements')

    # GPU requirements
    gpu_reqs = {
        'linux': [
            "tensorflow-gpu>=2.10.0",
            "cudatoolkit>=11.2",
            "cudnn>=8.1.0"
        ],
        'windows': [
            "tensorflow-gpu>=2.10.0",
            "cudatoolkit>=11.2",
            "cudnn>=8.1.0"
        ]
    }

    # CPU requirements
    cpu_reqs = {
        'linux': ["tensorflow-cpu>=2.10.0"],
        'windows': ["tensorflow-cpu>=2.10.0"],
        'macos-intel': ["tensorflow-macos>=2.10.0"],
        'macos-arm': ["tensorflow-macos>=2.10.0", "tensorflow-metal>=0.6.0"]
    }

    # Create each requirements file
    for platform_type, reqs in {**gpu_reqs, **cpu_reqs}.items():
        filename = f'requirements-{platform_type}.txt'
        with open(requirements_dir / filename, 'w') as f:
            f.write('\n'.join(reqs))


def create_dev_requirements():
    """Create development requirements file."""
    dev_reqs = [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "isort>=5.10.0",
        "pre-commit>=2.20.0"
    ]

    with open('./requirements/requirements-dev.txt', 'w') as f:
        f.write('\n'.join(dev_reqs))


def install_requirements():
    """Install requirements for the current platform."""
    try:
        # Create requirements directory and files
        Path('./requirements').mkdir(exist_ok=True)
        create_base_requirements()
        create_platform_requirements()
        create_dev_requirements()

        # Get appropriate requirements files
        base_reqs, platform_reqs = get_platform_requirements()

        print(f"Installing base requirements from {base_reqs}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(base_reqs)], check=True)

        print(f"Installing platform-specific requirements from {platform_reqs}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(platform_reqs)], check=True)

        # Install development requirements if specified
        if "--dev" in sys.argv:
            print("Installing development requirements")
            dev_reqs = Path('./requirements/requirements-dev.txt')
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(dev_reqs)], check=True)

            # Install pre-commit hooks
            subprocess.run(["pre-commit", "install"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)


def setup_directories():
    """Create necessary directories for NeuralMark."""
    # Base directories
    base_dirs = [
        'output',  # Base output directory
        'downloads',  # Downloaded images
        'logs',  # Log files
        'database'  # Global database (if not run-specific)
    ]

    for directory in base_dirs:
        Path(directory).mkdir(exist_ok=True)

    # Create a sample run directory structure for documentation
    sample_run = Path('output/sample_run_structure')
    run_dirs = [
        'fingerprinted',
        'manipulated',
        'plots',
        'reports',
        'test'
    ]

    for directory in run_dirs:
        dir_path = sample_run / directory
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create algorithm-specific subdirectories where needed
        if directory in ['fingerprinted', 'manipulated', 'plots', 'reports']:
            (dir_path / 'amdf').mkdir(exist_ok=True)
            (dir_path / 'cdha').mkdir(exist_ok=True)

    # Create .gitkeep files to preserve empty directories
    for path in Path('.').rglob('**/'):
        if path.is_dir() and not any(path.iterdir()):
            (path / '.gitkeep').touch()

if __name__ == "__main__":
    print("Setting up NeuralMark...")
    install_requirements()
    setup_directories()
    print("Setup completed successfully!")