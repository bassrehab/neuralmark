import unittest
from unittest.mock import patch
from setup import get_platform_requirements


class TestSetup(unittest.TestCase):
    @patch("platform.system")
    @patch("platform.machine")
    def test_get_platform_requirements_macos_arm(self, mock_machine, mock_system):
        """Test requirements selection for macOS ARM architecture."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        requirements_file = get_platform_requirements()
        self.assertEqual(requirements_file, "./requirements/requirements-macos-arm.txt")

    @patch("platform.system")
    @patch("platform.machine")
    def test_get_platform_requirements_linux_gpu(self, mock_machine, mock_system):
        """Test requirements selection for Linux with NVIDIA GPU."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        # Mock subprocess to simulate NVIDIA GPU presence
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            requirements_file = get_platform_requirements()
            self.assertTrue("requirements-linux-nvidia" in requirements_file)

    @patch("platform.system")
    @patch("platform.machine")
    def test_get_platform_requirements_windows(self, mock_machine, mock_system):
        """Test requirements selection for Windows."""
        mock_system.return_value = "Windows"
        mock_machine.return_value = "x86_64"
        requirements_file = get_platform_requirements()
        self.assertTrue("requirements-windows" in requirements_file)


if __name__ == "__main__":
    unittest.main()
