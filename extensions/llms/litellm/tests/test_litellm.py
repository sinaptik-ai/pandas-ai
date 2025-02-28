import os
import unittest
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import AuthenticationError

from extensions.llms.litellm.pandasai_litellm.litellm import LiteLLM
from pandasai.core.prompts.base import BasePrompt


class TestPrompt(BasePrompt):
    """Represents a test prompt with a customizable message template.

    This class extends the BasePrompt and provides a specific template
    for generating prompts. The template is defined as a simple string
    that includes a placeholder for a message.

    Attributes:
        template (str): The template string containing a placeholder
                        for the message to be inserted.

    Args:
        message (str): The message to be formatted into the template.

    Returns:
        str: The formatted prompt message based on the template."""

    template = "{{ message }}"


@pytest.fixture
def prompt():
    """Fixture that provides a test prompt instance.

    This fixture creates and returns a TestPrompt object initialized
    with a predefined message. It can be used in tests to simulate
    user input or interactions with the prompt.

    Returns:
        TestPrompt: An instance of TestPrompt with a message
        "Hello, how are you?"."""
    return TestPrompt(message="Hello, how are you?")


@pytest.fixture
def llm():
    """Fixture that provides an instance of LiteLLM configured with the GPT-3.5 Turbo model.

    This fixture can be used in tests to access a pre-initialized language model
    instance, facilitating testing of functionalities that require language model
    interactions.

    Returns:
        LiteLLM: An instance of LiteLLM initialized with the GPT-3.5 Turbo model."""
    return LiteLLM(model="gpt-3.5-turbo")


@patch("os.environ", {})
def test_missing_api_key(llm, prompt):
    """Tests the behavior of the API client when the API key is missing.

    This test verifies that an AuthenticationError is raised with the
    appropriate message when the API key is not set in the environment
    variables and an attempt is made to call the API with a prompt.

    Args:
        llm: The language model client being tested.
        prompt: The input prompt to be passed to the language model.

    Raises:
        AuthenticationError: If the API key is not provided in the environment."""
    with pytest.raises(
        AuthenticationError, match="The api_key client option must be set"
    ):
        llm.call(prompt)


@patch("os.environ", {"OPENAI_API_KEY": "key"})
def test_invalid_api_key(llm, prompt):
    """Tests the behavior of the language model when provided with an invalid API key.

    This test simulates the scenario where an incorrect OpenAI API key is set in the environment.
    It checks that the `llm.call` method raises an `AuthenticationError` with the expected error message.

    Args:
        llm: The language model instance used for making API calls.
        prompt: The input prompt to be sent to the language model.

    Raises:
        AuthenticationError: If the API key is invalid, indicating authentication failure."""
    with pytest.raises(AuthenticationError, match="Incorrect API key provided"):
        llm.call(prompt)


@patch("os.environ", {"OPENAI_API_KEY": "key"})
def test_successful_completion(llm, prompt):
    """Test the successful completion of a language model response.

    This function tests the behavior of a language model (LLM) when provided
    with a specific prompt. It mocks the completion function of the litellm
    library to provide a controlled response, allowing verification of the
    LLM's output and the parameters used in the completion call.

    Args:
        llm: The language model instance to test.
        prompt: The input prompt for the language model, typically a user message.

    Returns:
        None: This function asserts conditions and does not return a value.

    This test ensures that the LLM correctly processes the input prompt and
    returns the expected response while validating that the completion function
    was called with the appropriate arguments."""

    # Mock the litellm.completion function
    with patch(
        "extensions.llms.litellm.pandasai_litellm.litellm.completion"
    ) as completion_patch:
        # Create a mock response structure that matches litellm's response format
        mock_message = MagicMock()
        mock_message.content = "I'm doing well, thank you!"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Set the return value for the mocked completion function
        completion_patch.return_value = mock_response

        # Make the call
        response = llm.call(prompt)

        # Verify response
        assert response == "I'm doing well, thank you!"

        # Verify completion was called with correct parameters
        completion_patch.assert_called_once()
        args, kwargs = completion_patch.call_args

        # Ensure 'messages' was passed as expected
        assert kwargs["messages"] == [
            {"content": "Hello, how are you?", "role": "user"}
        ]
        assert kwargs["model"] == "gpt-3.5-turbo"


@patch("os.environ", {"OPENAI_API_KEY": "key"})
def test_completion_with_extra_params(prompt):
    """Test the completion functionality of LiteLLM with extra parameters.

    This test verifies that the LiteLLM instance calls the completion function
    with the expected parameters when provided with a prompt. It uses mocking
    to simulate the completion response and checks if the extra parameters
    are correctly passed.

    Args:
        prompt (str): The input prompt for the completion function.

    Returns:
        None"""
    # Create an instance of LiteLLM
    llm = LiteLLM(model="gpt-3.5-turbo", extra_param=10)

    # Mock the litellm.completion function
    with patch(
        "extensions.llms.litellm.pandasai_litellm.litellm.completion"
    ) as completion_patch:
        mock_message = MagicMock()
        mock_message.content = "I'm doing well, thank you!"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Set the return value for the mocked completion function
        completion_patch.return_value = mock_response

        llm.call(prompt)

        # Verify completion was called with correct parameters
        completion_patch.assert_called_once()
        args, kwargs = completion_patch.call_args

        assert kwargs["extra_param"] == 10
