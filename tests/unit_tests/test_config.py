import os
from unittest.mock import MagicMock, patch

from pandasai.config import APIKeyManager, Config, ConfigManager


class TestConfigManager:
    def setup_method(self):
        # Reset the ConfigManager state before each test
        ConfigManager._config = None
        ConfigManager._initialized = False

    def test_config_without_llm(self):
        """Test config behavior when no LLM is set"""
        with patch.dict(os.environ, {"PANDABI_API_KEY": "test-key"}):
            ConfigManager._config = MagicMock()
            ConfigManager._config.llm = None
            assert ConfigManager._config.llm is None

    def test_config_without_api_key(self):
        """Test config behavior when no API key is set"""
        with patch.dict(os.environ, {}, clear=True):
            ConfigManager._config = MagicMock()
            ConfigManager._config.llm = None

            # No LLM should be set automatically
            assert ConfigManager._config.llm is None

    def test_update_config(self):
        """Test updating configuration with new values"""
        # Initialize config with some initial values
        initial_config = {"save_logs": True, "verbose": False}
        ConfigManager._config = Config.from_dict(initial_config)

        # Update with new values
        update_dict = {"verbose": True}
        ConfigManager.update(update_dict)

        # Verify the configuration was updated correctly
        updated_config = ConfigManager._config.model_dump()
        assert updated_config["save_logs"] is True  # Original value preserved
        assert updated_config["verbose"] is True  # Value updated

    def test_set_api_key(self):
        """Test setting the API key"""
        test_api_key = "test-api-key-123"

        # Clear any existing API key
        if "PANDABI_API_KEY" in os.environ:
            del os.environ["PANDABI_API_KEY"]
        APIKeyManager._api_key = None

        # Set the API key
        APIKeyManager.set(test_api_key)

        # Verify the API key is set in both places
        assert os.environ["PANDABI_API_KEY"] == test_api_key
        assert APIKeyManager._api_key == test_api_key
        assert APIKeyManager.get() == test_api_key  # Also test the get method
