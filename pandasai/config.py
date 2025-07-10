import os
from importlib.util import find_spec
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from pandasai.helpers.filemanager import DefaultFileManager, FileManager
from pandasai.llm.base import LLM


class Config(BaseModel):
    save_logs: bool = True
    verbose: bool = False
    max_retries: int = 3
    llm: Optional[LLM] = None
    file_manager: FileManager = DefaultFileManager()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Config":
        return cls(**config)


class ConfigManager:
    """A singleton class to manage the global configuration."""

    _config: Config = Config()

    @classmethod
    def set(cls, config_dict: Dict[str, Any]) -> None:
        """Set the global configuration."""
        cls._config = Config.from_dict(config_dict)

    @classmethod
    def get(cls) -> Config:
        """Get the global configuration."""
        if cls._config is None:
            cls._config = Config()

        return cls._config

    @classmethod
    def update(cls, config_dict: Dict[str, Any]) -> None:
        """Update the existing configuration with new values."""
        current_config = cls._config.model_dump()
        current_config.update(config_dict)
        cls._config = Config.from_dict(current_config)


class APIKeyManager:
    _api_key: Optional[str] = None

    @classmethod
    def set(cls, api_key: str):
        os.environ["PANDABI_API_KEY"] = api_key
        cls._api_key = api_key

    @classmethod
    def get(cls) -> Optional[str]:
        return cls._api_key
