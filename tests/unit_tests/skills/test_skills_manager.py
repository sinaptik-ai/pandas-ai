"""
Tests for the SkillsManager class.
"""

from unittest.mock import MagicMock

import pytest

from pandasai.ee.skills import SkillType, skill
from pandasai.ee.skills.manager import SkillsManager


class TestSkillsManager:
    """Test cases for the SkillsManager class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing skills
        SkillsManager.clear_skills()

    def test_initial_state(self):
        """Test the initial state of SkillsManager."""
        assert len(SkillsManager.get_skills()) == 0
        assert not SkillsManager.has_skills()

    def test_add_single_skill(self):
        """Test adding a single skill."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)
        SkillsManager.add_skills(skill)

        assert len(SkillsManager.get_skills()) == 1
        assert SkillsManager.has_skills()
        assert SkillsManager.get_skills()[0].name == "test_function"

    def test_add_multiple_skills(self):
        """Test adding multiple skills at once."""

        def function1():
            """First function."""
            return "Hello"

        def function2():
            """Second function."""
            return "World"

        skill1 = SkillType(function1)
        skill2 = SkillType(function2)
        SkillsManager.add_skills(skill1, skill2)

        assert len(SkillsManager.get_skills()) == 2
        assert SkillsManager.has_skills()

        skill_names = [s.name for s in SkillsManager.get_skills()]
        assert "function1" in skill_names
        assert "function2" in skill_names

    def test_add_duplicate_skill_raises_error(self):
        """Test that adding a skill with a duplicate name raises an error."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill1 = SkillType(test_function)
        skill2 = SkillType(test_function, name="test_function")  # Same name

        SkillsManager.add_skills(skill1)

        with pytest.raises(
            ValueError, match="Skill with name 'test_function' already exists"
        ):
            SkillsManager.add_skills(skill2)

    def test_skill_exists(self):
        """Test checking if a skill exists."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)
        SkillsManager.add_skills(skill)

        assert SkillsManager.skill_exists("test_function")
        assert not SkillsManager.skill_exists("nonexistent_function")

    def test_get_skill_by_func_name(self):
        """Test getting a skill by its function name."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)
        SkillsManager.add_skills(skill)

        retrieved_skill = SkillsManager.get_skill_by_func_name("test_function")
        assert retrieved_skill is not None
        assert retrieved_skill.name == "test_function"
        assert retrieved_skill.func == test_function

        # Test getting non-existent skill
        retrieved_skill = SkillsManager.get_skill_by_func_name("nonexistent")
        assert retrieved_skill is None

    def test_get_skills_returns_copy(self):
        """Test that get_skills returns a copy, not the original list."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)
        SkillsManager.add_skills(skill)

        skills_copy = SkillsManager.get_skills()
        skills_copy.append("not_a_skill")  # This should not affect the original

        original_skills = SkillsManager.get_skills()
        assert len(original_skills) == 1
        assert isinstance(original_skills[0], SkillType)

    def test_clear_skills(self):
        """Test clearing all skills."""

        def function1():
            """First function."""
            return "Hello"

        def function2():
            """Second function."""
            return "World"

        skill1 = SkillType(function1)
        skill2 = SkillType(function2)
        SkillsManager.add_skills(skill1, skill2)

        assert len(SkillsManager.get_skills()) == 2

        SkillsManager.clear_skills()

        assert len(SkillsManager.get_skills()) == 0
        assert not SkillsManager.has_skills()

    def test_string_representation(self):
        """Test the string representation of SkillsManager."""

        def function1():
            """First function."""
            return "Hello"

        def function2():
            """Second function."""
            return "World"

        skill1 = SkillType(function1)
        skill2 = SkillType(function2)
        SkillsManager.add_skills(skill1, skill2)

        skills_str = SkillsManager.__str__()

        # Should contain both function definitions
        assert "def function1():" in skills_str
        assert "def function2():" in skills_str
        assert "First function." in skills_str
        assert "Second function." in skills_str

    def test_global_state_persistence(self):
        """Test that SkillsManager maintains global state across instances."""

        def test_function():
            """A test function."""
            return "Hello, world!"

        skill = SkillType(test_function)
        SkillsManager.add_skills(skill)

        # Create a new instance (simulating different parts of the application)
        from pandasai.ee.skills.manager import SkillsManager as NewSkillsManager

        # The new instance should see the same skills
        assert len(NewSkillsManager.get_skills()) == 1
        assert NewSkillsManager.skill_exists("test_function")
        assert NewSkillsManager.has_skills()
