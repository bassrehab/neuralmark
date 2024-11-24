import platform
import subprocess
import sys


def get_platform_requirements():
    """Determine the appropriate requirements file based on platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":  # macOS
        if "arm" in machine:
            return "./requirements/requirements-macos-arm.txt"
        else:
            return "./requirements/requirements-macos-intel.txt"

    elif system == "linux":
        # Check for NVIDIA GPU
        try:
            subprocess.run(["nvidia-smi"], capture_output=True)
            return "./requirements/requirements-linux-gpu.txt"
        except FileNotFoundError:
            return "./requirements/requirements-linux-cpu.txt"

    elif system == "windows":
        # Check for NVIDIA GPU
        try:
            subprocess.run(["nvidia-smi"], capture_output=True)
            return "./requirements/requirements-windows-gpu.txt"
        except FileNotFoundError:
            return "./requirements/requirements-windows-cpu.txt"

    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)


def install_requirements():
    """Install requirements for the current platform."""
    requirements_file = get_platform_requirements()
    print(f"Installing requirements from {requirements_file}")

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file])

        # Install development requirements if specified
        if "--dev" in sys.argv:
            print("Installing development requirements")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "./requirements/requirements-dev.txt"])

        # Install documentation requirements if specified
        if "--docs" in sys.argv:
            print("Installing documentation requirements")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "./requirements/requirements-docs.txt"])

    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_requirements()
