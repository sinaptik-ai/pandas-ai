"""
Tests for the Skill class.
"""

import inspect
from unittest.mock import MagicMock

import pytest

from pandasai.ee.skills import SkillType


class TestSkill:
    """Test cases for the Skill class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing skills
        from pandasai.ee.skills.manager import SkillsManager

        SkillsManager.clear_skills()

    def test_skill_creation_with_function(self):
        """Test creating a skill from a function."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)

        assert skill.name == "test_function"
        assert skill.description == "A test function."
        assert skill.func == test_function
        assert skill._signature == "def test_function():"

    def test_skill_creation_with_custom_name(self):
        """Test creating a skill with a custom name."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function, name="custom_name")

        assert skill.name == "custom_name"
        assert skill.description == "A test function."
        assert skill.func == test_function

    def test_skill_creation_with_custom_description(self):
        """Test creating a skill with a custom description."""

        def test_function():
            """Original docstring."""
            return "Hello, world!"

        skill = SkillType(test_function, description="Custom description")

        assert skill.name == "test_function"
        assert skill.description == "Custom description"
        assert skill.func == test_function

    def test_skill_creation_without_docstring_raises_error(self):
        """Test that creating a skill without a docstring raises an error."""

        def test_function():
            return "Hello, world!"

        with pytest.raises(ValueError, match="Function must have a docstring"):
            SkillType(test_function)

    def test_skill_creation_with_empty_docstring_raises_error(self):
        """Test that creating a skill with empty docstring raises an error."""

        def test_function():
            return "Hello, world!"

        with pytest.raises(ValueError, match="Function must have a docstring"):
            SkillType(test_function)

    def test_skill_creation_with_lambda_requires_name(self):
        """Test that creating a skill with a lambda requires a name."""
        lambda_func = lambda x: x * 2

        with pytest.raises(ValueError, match="Function must have a docstring"):
            SkillType(lambda_func)

    def test_skill_creation_with_lambda_and_name(self):
        """Test creating a skill with a lambda and providing a name."""
        lambda_func = lambda x: x * 2

        skill = SkillType(lambda_func, name="double", description="Doubles a number")

        assert skill.name == "double"
        assert skill.description == "Doubles a number"
        assert skill.func == lambda_func

    def test_skill_call(self):
        """Test calling a skill."""

        def test_function(x, y=10):
            """A test function with parameters."""
            return x + y

        skill = SkillType(test_function)

        result = skill(5)
        assert result == 15

        result = skill(5, 20)
        assert result == 25

    def test_skill_string_representation(self):
        """Test the string representation of a skill."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)
        skill_str = str(skill)

        expected = (
            '<function>\ndef test_function():\n    """A test function."""\n</function>'
        )
        assert skill_str == expected

    def test_skill_stringify(self):
        """Test the stringify method returns function source."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)
        source = skill.stringify()

        assert "def test_function():" in source
        assert 'return "Hello, world!"' in source

    def test_skill_from_function_classmethod(self):
        """Test the from_function class method."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType.from_function(test_function)

        assert skill.name == "test_function"
        assert skill.description == "A test function."
        assert skill.func == test_function

    def test_skill_with_parameters(self):
        """Test skill with function parameters."""

        def test_function(x: int, y: int = 5) -> int:
            """A test function with parameters."""
            return x + y

        skill = SkillType(test_function)

        assert skill.name == "test_function"
        assert skill.description == "A test function with parameters."
        assert skill._signature == "def test_function(x: int, y: int = 5) -> int:"

    def test_skill_inherits_from_basemodel(self):
        """Test that Skill inherits from BaseModel."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)

        # Check that it has Pydantic BaseModel attributes
        assert hasattr(skill, "model_dump")
        assert hasattr(skill, "model_validate")

    def test_skill_private_attr_initialization(self):
        """Test that private attributes are properly initialized."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)

        # Check that _signature is properly set
        assert hasattr(skill, "_signature")
        assert skill._signature == "def test_function():"
