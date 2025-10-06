"""
Tests for the shared SQL functions template.
"""

import os
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from pandasai.ee.skills import skill
from pandasai.ee.skills.manager import SkillsManager


class TestSharedTemplate:
    """Test cases for the shared SQL functions template."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing skills
        SkillsManager.clear_skills()

    def get_template_environment(self):
        """Get the Jinja2 template environment."""
        current_dir = Path(__file__).parent
        template_path = (
            current_dir.parent.parent.parent
            / "pandasai"
            / "core"
            / "prompts"
            / "templates"
        )
        return Environment(loader=FileSystemLoader(str(template_path)))

    def test_shared_template_without_skills(self):
        """Test the shared template when no skills are present."""
        env = self.get_template_environment()
        template = env.get_template("shared/sql_functions.tmpl")

        # Mock context without skills
        class MockContext:
            def __init__(self):
                self.skills = []

        context = MockContext()
        rendered = template.render(context=context)

        # Should only contain execute_sql_query
        assert "execute_sql_query" in rendered
        assert "def execute_sql_query(sql_query: str) -> pd.DataFrame" in rendered
        assert "This method connects to the database" in rendered

        # Should not contain any custom skills
        assert "def hello_world():" not in rendered
        assert "def custom_function():" not in rendered

    def test_shared_template_with_skills(self):
        """Test the shared template when skills are present."""

        # Add some skills
        @skill
        def hello_world():
            """A simple greeting function."""
            return "Hello, world!"

        @skill("custom_function")
        def another_function():
            """A custom function."""
            return "Custom result"

        env = self.get_template_environment()
        template = env.get_template("shared/sql_functions.tmpl")

        # Mock context with skills
        class MockContext:
            def __init__(self):
                self.skills = SkillsManager.get_skills()

        context = MockContext()
        rendered = template.render(context=context)

        # Should contain execute_sql_query
        assert "execute_sql_query" in rendered
        assert "def execute_sql_query(sql_query: str) -> pd.DataFrame" in rendered

        # Should contain custom skills
        assert "def hello_world():" in rendered
        assert "def custom_function():" in rendered
        assert "A simple greeting function." in rendered
        assert "A custom function." in rendered

    def test_shared_template_formatting(self):
        """Test that the shared template has correct formatting."""

        @skill
        def test_function():
            """A test function."""
            return "test"

        env = self.get_template_environment()
        template = env.get_template("shared/sql_functions.tmpl")

        class MockContext:
            def __init__(self):
                self.skills = SkillsManager.get_skills()

        context = MockContext()
        rendered = template.render(context=context)

        # Check the structure
        lines = rendered.split("\n")

        # Should start with the header
        assert "The following functions have already been provided" in lines[0]
        assert "Please use them as needed and do not redefine them" in lines[0]

        # Should contain function blocks
        assert "<function>" in rendered
        assert "</function>" in rendered

        # Should not have extra newlines between functions
        # Check that there are no empty lines between function blocks
        function_blocks = rendered.split("<function>")
        for i, block in enumerate(function_blocks[1:], 1):  # Skip first empty block
            if i < len(function_blocks) - 1:  # Not the last block
                # Should not start with multiple newlines
                assert not block.startswith("\n\n")

    def test_shared_template_conditional_rendering(self):
        """Test that the shared template conditionally renders skills."""
        env = self.get_template_environment()
        template = env.get_template("shared/sql_functions.tmpl")

        # Test with empty skills list
        class MockContextEmpty:
            def __init__(self):
                self.skills = []

        context_empty = MockContextEmpty()
        rendered_empty = template.render(context=context_empty)

        # Should only have execute_sql_query
        function_count = rendered_empty.count("<function>")
        assert function_count == 1  # Only execute_sql_query

        # Test with skills
        @skill
        def test_function():
            """A test function."""
            return "test"

        class MockContextWithSkills:
            def __init__(self):
                self.skills = SkillsManager.get_skills()

        context_with_skills = MockContextWithSkills()
        rendered_with_skills = template.render(context=context_with_skills)

        # Should have execute_sql_query plus custom skills
        function_count = rendered_with_skills.count("<function>")
        assert function_count == 2  # execute_sql_query + test_function

    def test_shared_template_skill_string_formatting(self):
        """Test that skills are properly formatted in the template."""

        @skill
        def complex_function(x: int, y: str = "default") -> str:
            """A complex function with parameters."""
            return f"x={x}, y={y}"

        env = self.get_template_environment()
        template = env.get_template("shared/sql_functions.tmpl")

        class MockContext:
            def __init__(self):
                self.skills = SkillsManager.get_skills()

        context = MockContext()
        rendered = template.render(context=context)

        # Check that the complex function is properly formatted
        assert "def complex_function(x: int, y: str = 'default') -> str:" in rendered
        assert "A complex function with parameters." in rendered
        assert "<function>" in rendered
        assert "</function>" in rendered

    def test_shared_template_multiple_skills_order(self):
        """Test that multiple skills are rendered in the correct order."""

        @skill("first_function")
        def function1():
            """First function."""
            return "first"

        @skill("second_function")
        def function2():
            """Second function."""
            return "second"

        @skill("third_function")
        def function3():
            """Third function."""
            return "third"

        env = self.get_template_environment()
        template = env.get_template("shared/sql_functions.tmpl")

        class MockContext:
            def __init__(self):
                self.skills = SkillsManager.get_skills()

        context = MockContext()
        rendered = template.render(context=context)

        # Check that all functions are present
        assert "def first_function():" in rendered
        assert "def second_function():" in rendered
        assert "def third_function():" in rendered

        # Check that execute_sql_query comes first
        execute_pos = rendered.find("def execute_sql_query")
        first_pos = rendered.find("def first_function")
        second_pos = rendered.find("def second_function")
        third_pos = rendered.find("def third_function")

        assert execute_pos < first_pos
        assert first_pos < second_pos
        assert second_pos < third_pos

    def test_shared_template_no_extra_newlines(self):
        """Test that the shared template doesn't add extra newlines."""

        @skill
        def test_function():
            """A test function."""
            return "test"

        env = self.get_template_environment()
        template = env.get_template("shared/sql_functions.tmpl")

        class MockContext:
            def __init__(self):
                self.skills = SkillsManager.get_skills()

        context = MockContext()
        rendered = template.render(context=context)

        # Check for excessive newlines (more than 2 consecutive)
        lines = rendered.split("\n")
        consecutive_empty = 0
        max_consecutive_empty = 0

        for line in lines:
            if line.strip() == "":
                consecutive_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, consecutive_empty)
            else:
                consecutive_empty = 0

        # Should not have more than 2 consecutive empty lines
        assert max_consecutive_empty <= 2
