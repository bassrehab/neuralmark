import unittest
from alphapunch.utils.config import load_environment, load_config
import os


class TestConfig(unittest.TestCase):
    def test_load_environment(self):
        """Test loading environment variables."""
        os.environ["UNSPLASH_ACCESS_KEY"] = "test_access_key"
        os.environ["PRIVATE_KEY"] = "test_private_key"

        env = load_environment()
        self.assertEqual(env["UNSPLASH_ACCESS_KEY"], "test_access_key")
        self.assertEqual(env["PRIVATE_KEY"], "test_private_key")

    def test_load_config(self):
        """Test loading configuration from YAML file."""
        with open("test_config.yaml", "w") as f:
            f.write("key1: value1\nkey2: value2")

        config = load_config("test_config.yaml")
        self.assertIn("key1", config)
        self.assertEqual(config["key1"], "value1")
        self.assertIn("key2", config)
        self.assertEqual(config["key2"], "value2")

        # Clean up test file
        os.remove("test_config.yaml")


if __name__ == "__main__":
    unittest.main()
