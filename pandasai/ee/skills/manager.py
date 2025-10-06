from typing import List

from pandasai.ee.skills import SkillType


class SkillsManager:
    """
    A singleton class to manage the global skills list.
    """

    _skills: List[SkillType] = []

    @classmethod
    def add_skills(cls, *skills: SkillType):
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
    def get_skills(cls) -> List[SkillType]:
        """
        Get the global list of skills.

        Returns:
            List[SkillType]: The list of all skills.
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
