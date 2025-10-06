import unittest
from unittest.mock import MagicMock, patch

from pandasai.core.code_execution.environment import (
    get_environment,
    get_version,
    import_dependency,
)


class TestEnvironmentFunctions(unittest.TestCase):
    @patch("pandasai.core.code_execution.environment.import_dependency")
    def test_get_environment_with_secure_mode(self, mock_import_dependency):
        """Test get_environment function in secure mode."""
        mock_import_dependency.side_effect = lambda name: MagicMock(name=name)
        env = get_environment()

        self.assertIn("pd", env)
        self.assertIn("plt", env)
        self.assertIn("np", env)
        self.assertIn("px", env)

    @patch("pandasai.core.code_execution.environment.import_dependency")
    def test_get_environment_without_secure_mode(self, mock_import_dependency):
        """Test get_environment function in non-secure mode."""
        mock_import_dependency.side_effect = lambda name: MagicMock(name=name)
        env = get_environment()

        self.assertIn("pd", env)
        self.assertIn("plt", env)
        self.assertIn("np", env)
        self.assertIn("px", env)
        self.assertIsInstance(env["pd"], MagicMock)

    @patch("pandasai.core.code_execution.environment.importlib.import_module")
    def test_import_dependency_success(self, mock_import_module):
        """Test successful import of a dependency."""
        mock_import_module.return_value = MagicMock(__version__="1.0.0")
        module = import_dependency("numpy")

        self.assertIsNotNone(module)

    @patch("pandasai.core.code_execution.environment.importlib.import_module")
    def test_import_dependency_missing(self, mock_import_module):
        """Test handling of a missing dependency."""
        mock_import_module.side_effect = ImportError("Module not found")
        with self.assertRaises(ImportError):
            import_dependency("non_existent_module")

    @patch("pandasai.core.code_execution.environment.importlib.import_module")
    def test_import_dependency_with_extra_message(self, mock_import_module):
        """Test import dependency with additional error message."""
        mock_import_module.side_effect = ImportError("Module not found")
        with self.assertRaises(ImportError) as context:
            import_dependency("non_existent_module", extra="Please install it.")

        self.assertIn("Please install it.", str(context.exception))

    @patch("pandasai.core.code_execution.environment.importlib.import_module")
    def test_get_version_success(self, mock_import_module):
        """Test getting the version of a module successfully."""
        mock_import_module.return_value = MagicMock(__version__="1.0.0")
        version = get_version(mock_import_module("numpy"))
        self.assertEqual(version, "1.0.0")

    @patch("pandasai.core.code_execution.environment.importlib.import_module")
    def test_get_version_failure(self, mock_import_module):
        """Test getting version fails when __version__ is not present."""
        module_mock = MagicMock()
        module_mock.__name__ = "numpy"
        mock_import_module.return_value = module_mock
        with self.assertRaises(ImportError):
            get_version(mock_import_module("numpy"))


if __name__ == "__main__":
    unittest.main()
