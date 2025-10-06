from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pandasai.config import Config, ConfigManager
from pandasai.constants import DEFAULT_CHART_DIRECTORY
from pandasai.data_loader.semantic_layer_schema import is_schema_source_same
from pandasai.ee.skills.manager import SkillsManager
from pandasai.exceptions import InvalidConfigError
from pandasai.helpers.folder import Folder
from pandasai.helpers.logger import Logger
from pandasai.helpers.memory import Memory
from pandasai.vectorstores.vectorstore import VectorStore

if TYPE_CHECKING:
    from pandasai.dataframe import DataFrame, VirtualDataFrame
    from pandasai.llm.base import LLM


@dataclass
class AgentState:
    """
    Context class for managing pipeline attributes and passing them between steps.
    """

    dfs: List[Union[DataFrame, VirtualDataFrame]] = field(default_factory=list)
    _config: Union[Config, dict] = field(default_factory=dict)
    memory: Memory = field(default_factory=Memory)
    vectorstore: Optional[VectorStore] = None
    intermediate_values: Dict[str, Any] = field(default_factory=dict)
    logger: Optional[Logger] = None
    last_code_generated: Optional[str] = None
    last_code_executed: Optional[str] = None
    last_prompt_id: str = None
    last_prompt_used: str = None
    output_type: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = Config(**self.config)

    def initialize(
        self,
        dfs: Union[
            Union[DataFrame, VirtualDataFrame], List[Union[DataFrame, VirtualDataFrame]]
        ],
        config: Optional[Union[Config, dict]] = None,
        memory_size: Optional[int] = 10,
        vectorstore: Optional[VectorStore] = None,
        description: str = None,
    ):
        """Initialize the state with the given parameters."""
        self.dfs = dfs if isinstance(dfs, list) else [dfs]
        self.config = self._get_config(config)
        self.skills = SkillsManager.get_skills()
        if config:
            self.config.llm = self._get_llm(self.config.llm)
        self.memory = Memory(memory_size, agent_description=description)
        self.logger = Logger(
            save_logs=self.config.save_logs, verbose=self.config.verbose
        )
        self.vectorstore = vectorstore
        self._configure()

    def _configure(self):
        """Configure paths for charts."""
        # Add project root path if save_charts_path is default
        Folder.create(DEFAULT_CHART_DIRECTORY)

    def _get_config(self, config: Union[Config, dict, None]) -> Config:
        """Load a config to be used for queries."""
        if config is None:
            return ConfigManager.get()

        if isinstance(config, dict):
            return Config(**config)

        return config

    def _get_llm(self, llm: Optional[LLM] = None) -> LLM:
        """Load and configure the LLM."""
        return llm

    def assign_prompt_id(self):
        """Assign a new prompt ID."""
        self.last_prompt_id = uuid.uuid4()

        if self.logger:
            self.logger.log(f"Prompt ID: {self.last_prompt_id}")

    def reset_intermediate_values(self):
        """Resets the intermediate values dictionary."""
        self.intermediate_values.clear()

    def add(self, key: str, value: Any):
        """Adds a single key-value pair to intermediate values."""
        self.intermediate_values[key] = value

    def add_many(self, values: Dict[str, Any]):
        """Adds multiple key-value pairs to intermediate values."""
        self.intermediate_values.update(values)

    def get(self, key: str, default: Any = "") -> Any:
        """Fetches a value from intermediate values or returns a default."""
        return self.intermediate_values.get(key, default)

    @property
    def config(self):
        """
        Returns the local config if set, otherwise fetches the global config.
        """
        if self._config is not None:
            return self._config

        import pandasai as pai

        return pai.config.get()

    @config.setter
    def config(self, value: Union[Config, dict, None]):
        """
        Allows setting a new config value.
        """
        self._config = Config(**value) if isinstance(value, dict) else value
