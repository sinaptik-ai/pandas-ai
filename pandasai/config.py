import os
from importlib.util import find_spec
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from pandasai.helpers.filemanager import DefaultFileManager, FileManager
from pandasai.llm.base import LLM
from pandasai.skills import Skill


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


class SkillsManager:
    """
    A singleton class to manage the global skills list.
    """

    _skills: List[Skill] = []

    @classmethod
    def add_skills(cls, *skills: Skill):
        """
        Add skills to the global list of skills. If a skill with the same name
             already exists, raise an error.

        Args:
            *skills: Variable number of skill objects to add.
        """
        for skill in skills:
            if any(existing_skill.name == skill.name for existing_skill in cls._skills):
                raise ValueError(f"Skill with name '{skill.name}' already exists.")

        cls._skills.extend(skills)

    @classmethod
    def skill_exists(cls, name: str):
        """
        Check if a skill with the given name exists in the global list of skills.

        Args:
            name (str): The name of the skill to check.

        Returns:
            bool: True if a skill with the given name exists, False otherwise.
        """
        return any(skill.name == name for skill in cls._skills)

    @classmethod
    def has_skills(cls):
        """
        Check if there are any skills in the global list of skills.

        Returns:
            bool: True if there are skills, False otherwise.
        """
        return len(cls._skills) > 0

    @classmethod
    def get_skill_by_func_name(cls, name: str):
        """
        Get a skill by its name from the global list.

        Args:
            name (str): The name of the skill to retrieve.

        Returns:
            Skill or None: The skill with the given name, or None if not found.
        """
        return next((skill for skill in cls._skills if skill.name == name), None)

    @classmethod
    def get_skills(cls) -> List[Skill]:
        """
        Get the global list of skills.

        Returns:
            List[Skill]: The list of all skills.
        """
        return cls._skills.copy()

    @classmethod
    def clear_skills(cls):
        """
        Clear all skills from the global list.
        """
        cls._skills.clear()

    @classmethod
    def __str__(cls) -> str:
        """
        Present all skills
        Returns:
            str: String representation of all skills
        """
        return "\n".join(str(skill) for skill in cls._skills)
