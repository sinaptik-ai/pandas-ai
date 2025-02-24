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
        cls.validate_llm()

    @classmethod
    def get(cls) -> Config:
        """Get the global configuration."""
        if cls._config is None:
            cls._config = Config()

        if cls._config.llm is None and os.environ.get("PANDABI_API_KEY"):
            from pandasai.llm.bamboo_llm import BambooLLM

            cls._config.llm = BambooLLM()

        return cls._config

    @classmethod
    def update(cls, config_dict: Dict[str, Any]) -> None:
        """Update the existing configuration with new values."""
        current_config = cls._config.model_dump()
        current_config.update(config_dict)
        cls._config = Config.from_dict(current_config)

    @classmethod
    def validate_llm(cls):
        """
        Initializes a default LLM if not provided.
        """
        if cls._config.llm is None and os.environ.get("PANDABI_API_KEY"):
            from pandasai.llm.bamboo_llm import BambooLLM

            cls._config.llm = BambooLLM()
            return

        # Check if pandasai_langchain is installed
        if find_spec("pandasai_langchain") is not None:
            from pandasai_langchain.langchain import LangchainLLM, is_langchain_llm

            if is_langchain_llm(cls._config.llm):
                cls._config.llm = LangchainLLM(cls._config.llm)


class APIKeyManager:
    _api_key: Optional[str] = None

    @classmethod
    def set(cls, api_key: str):
        os.environ["PANDABI_API_KEY"] = api_key
        cls._api_key = api_key

    @classmethod
    def get(cls) -> Optional[str]:
        return cls._api_key
