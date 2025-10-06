"""
Tests for the skill decorator.
"""

from unittest.mock import MagicMock, patch

import pytest

from pandasai.ee.skills import SkillType, skill
from pandasai.ee.skills.manager import SkillsManager

# Alias for backward compatibility in tests
Skill = SkillType


class TestSkillDecorator:
    """Test cases for the skill decorator."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing skills
        SkillsManager.clear_skills()

    def test_skill_decorator_without_arguments(self):
        """Test using the skill decorator without arguments."""

        @skill
        def test_function():
            """A test function."""
            return "Hello, world!"

        # Check that the function is now a Skill object
        assert isinstance(test_function, Skill)
        assert test_function.name == "test_function"
        assert test_function.description == "A test function."

        # Check that the skill was automatically added to SkillsManager
        skills = SkillsManager.get_skills()
        assert len(skills) == 1
        assert skills[0].name == "test_function"

    def test_skill_decorator_with_custom_name(self):
        """Test using the skill decorator with a custom name."""

        @skill("custom_name")
        def test_function():
            """A test function."""
            return "Hello, world!"

        # Check that the function is now a Skill object
        assert isinstance(test_function, Skill)
        assert test_function.name == "custom_name"
        assert test_function.description == "A test function."

        # Check that the skill was automatically added to SkillsManager
        skills = SkillsManager.get_skills()
        assert len(skills) == 1
        assert skills[0].name == "custom_name"

    def test_skill_decorator_with_parentheses(self):
        """Test using the skill decorator with parentheses."""

        @skill()
        def test_function():
            """A test function."""
            return "Hello, world!"

        # Check that the function is now a Skill object
        assert isinstance(test_function, Skill)
        assert test_function.name == "test_function"
        assert test_function.description == "A test function."

        # Check that the skill was automatically added to SkillsManager
        skills = SkillsManager.get_skills()
        assert len(skills) == 1
        assert skills[0].name == "test_function"

    def test_skill_decorator_multiple_skills(self):
        """Test using the skill decorator multiple times."""

        @skill
        def function1():
            """First function."""
            return "Hello"

        @skill("custom_name")
        def function2():
            """Second function."""
            return "World"

        @skill()
        def function3():
            """Third function."""
            return "!"

        # Check that all functions are Skill objects
        assert isinstance(function1, Skill)
        assert isinstance(function2, Skill)
        assert isinstance(function3, Skill)

        # Check that all skills were automatically added to SkillsManager
        skills = SkillsManager.get_skills()
        assert len(skills) == 3

        skill_names = [s.name for s in skills]
        assert "function1" in skill_names
        assert "custom_name" in skill_names
        assert "function3" in skill_names

    def test_skill_decorator_with_parameters(self):
        """Test using the skill decorator with a function that has parameters."""

        @skill
        def test_function(x: int, y: int = 5) -> int:
            """A test function with parameters."""
            return x + y

        # Check that the function is now a Skill object
        assert isinstance(test_function, Skill)
        assert test_function.name == "test_function"
        assert test_function.description == "A test function with parameters."
        assert (
            test_function._signature == "def test_function(x: int, y: int = 5) -> int:"
        )

    def test_skill_decorator_calling_function(self):
        """Test that the decorated function can still be called."""

        @skill
        def test_function(x: int) -> int:
            """A test function."""
            return x * 2

        # Check that the function can still be called
        result = test_function(5)
        assert result == 10

    def test_skill_decorator_without_docstring_raises_error(self):
        """Test that the skill decorator raises an error for functions without docstrings."""
        with pytest.raises(ValueError, match="Function must have a docstring"):

            @skill
            def test_function():
                return "Hello, world!"

    def test_skill_decorator_too_many_arguments_raises_error(self):
        """Test that the skill decorator raises an error with too many arguments."""
        with pytest.raises(ValueError, match="Too many arguments for skill decorator"):

            @skill("name1", "name2")
            def test_function():
                """A test function."""
                return "Hello, world!"

    def test_skill_decorator_duplicate_names_raises_error(self):
        """Test that adding skills with duplicate names raises an error."""

        @skill("duplicate_name")
        def function1():
            """First function."""
            return "Hello"

        # This should raise an error because the name already exists
        with pytest.raises(
            ValueError, match="Skill with name 'duplicate_name' already exists"
        ):

            @skill("duplicate_name")
            def function2():
                """Second function."""
                return "World"

    def test_skill_decorator_string_representation(self):
        """Test the string representation of decorated skills."""

        @skill
        def test_function():
            """A test function."""
            return "Hello, world!"

        skill_str = str(test_function)
        expected = (
            '<function>\ndef test_function():\n    """A test function."""\n</function>'
        )
        assert skill_str == expected

    def test_skill_decorator_stringify(self):
        """Test the stringify method of decorated skills."""

        @skill
        def test_function():
            """A test function."""
            return "Hello, world!"

        source = test_function.stringify()
        assert "def test_function():" in source
        assert 'return "Hello, world!"' in source
