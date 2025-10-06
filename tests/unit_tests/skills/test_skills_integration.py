"""
Integration tests for the skills system.
"""

from unittest.mock import MagicMock, patch

import pytest

from pandasai.agent.state import AgentState
from pandasai.ee.skills import SkillType, skill
from pandasai.ee.skills.manager import SkillsManager

# Alias for backward compatibility in tests
Skill = SkillType


class TestSkillsIntegration:
    """Integration tests for the skills system."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing skills
        SkillsManager.clear_skills()

    def test_skill_decorator_auto_registration(self):
        """Test that the skill decorator automatically registers skills."""

        @skill
        def test_function():
            """A test function."""
            return "Hello, world!"

        # Check that the skill was automatically registered
        assert len(SkillsManager.get_skills()) == 1
        assert SkillsManager.skill_exists("test_function")

        # Check that the function is now a Skill object
        assert isinstance(test_function, SkillType)
        assert test_function.name == "test_function"

    def test_agent_state_includes_skills(self):
        """Test that AgentState includes skills from SkillsManager."""

        @skill
        def test_function():
            """A test function."""
            return "Hello, world!"

        @skill("custom_name")
        def another_function():
            """Another function."""
            return "Another result"

        # Create a mock AgentState
        state = AgentState()

        # Mock the initialization to avoid full setup
        with patch.object(state, "_get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            state.initialize([], config=None, memory_size=10)

            # Check that skills are included in the state
            assert len(state.skills) == 2
            skill_names = [s.name for s in state.skills]
            assert "test_function" in skill_names
            assert "custom_name" in skill_names

    def test_skills_available_in_templates(self):
        """Test that skills are available in template rendering."""

        @skill
        def test_function():
            """A test function."""
            return "Hello, world!"

        @skill("custom_name")
        def another_function():
            """Another function."""
            return "Another result"

        # Create a mock context with skills
        class MockContext:
            def __init__(self):
                self.skills = SkillsManager.get_skills()

        context = MockContext()

        # Test template rendering logic
        if context.skills:
            skill_strings = [str(skill) for skill in context.skills]

            # Check that both skills are rendered
            assert len(skill_strings) == 2
            assert any("def test_function():" in s for s in skill_strings)
            assert any("def custom_name():" in s for s in skill_strings)

    def test_skills_work_with_different_function_signatures(self):
        """Test that skills work with different function signatures."""

        @skill
        def simple_function():
            """A simple function."""
            return "simple"

        @skill
        def function_with_params(x: int, y: int = 5) -> int:
            """A function with parameters."""
            return x + y

        @skill
        def function_with_args(*args, **kwargs):
            """A function with args and kwargs."""
            return len(args) + len(kwargs)

        # Check that all skills are registered
        assert len(SkillsManager.get_skills()) == 3
        assert SkillsManager.skill_exists("simple_function")
        assert SkillsManager.skill_exists("function_with_params")
        assert SkillsManager.skill_exists("function_with_args")

        # Check that all functions can still be called
        assert simple_function() == "simple"
        assert function_with_params(5) == 10
        assert function_with_params(5, 10) == 15
        assert function_with_args(1, 2, 3, a=1, b=2) == 5

    def test_skills_clear_and_rebuild(self):
        """Test clearing skills and rebuilding the system."""

        @skill
        def function1():
            """First function."""
            return "first"

        @skill
        def function2():
            """Second function."""
            return "second"

        # Check initial state
        assert len(SkillsManager.get_skills()) == 2

        # Clear skills
        SkillsManager.clear_skills()
        assert len(SkillsManager.get_skills()) == 0

        # Add new skills
        @skill
        def function3():
            """Third function."""
            return "third"

        @skill("new_name")
        def function4():
            """Fourth function."""
            return "fourth"

        # Check new state
        assert len(SkillsManager.get_skills()) == 2
        assert SkillsManager.skill_exists("function3")
        assert SkillsManager.skill_exists("new_name")

    def test_skills_with_complex_descriptions(self):
        """Test skills with complex docstrings."""

        @skill
        def complex_function(x: int, y: str = "default") -> str:
            """
            A complex function with detailed documentation.

            Args:
                x: An integer parameter
                y: A string parameter with default value

            Returns:
                A formatted string

            Example:
                >>> complex_function(5, "test")
                "x=5, y=test"
            """
            return f"x={x}, y={y}"

        skill_obj = SkillsManager.get_skill_by_func_name("complex_function")
        assert skill_obj is not None
        assert "A complex function with detailed documentation" in skill_obj.description
        assert (
            skill_obj._signature
            == "def complex_function(x: int, y: str = 'default') -> str:"
        )

    def test_skills_error_handling(self):
        """Test error handling in the skills system."""
        # Test function without docstring
        with pytest.raises(ValueError):

            @skill
            def no_docstring():
                return "no docstring"

        # Test duplicate names
        @skill("duplicate")
        def first_function():
            """First function."""
            return "first"

        with pytest.raises(ValueError, match="already exists"):

            @skill("duplicate")
            def second_function():
                """Second function."""
                return "second"
