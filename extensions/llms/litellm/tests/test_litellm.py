import os
import unittest
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import AuthenticationError

from extensions.llms.litellm.pandasai_litellm.litellm import LiteLLM
from pandasai.core.prompts.base import BasePrompt


class TestPrompt(BasePrompt):
    template = "{{ message }}"

@pytest.fixture
def prompt():
    return TestPrompt(message="Hello, how are you?")

@pytest.fixture
def llm():
    return LiteLLM(model="gpt-3.5-turbo")

@patch("os.environ", {})
def test_missing_api_key(llm, prompt):
    with pytest.raises(AuthenticationError,match="The api_key client option must be set"):
        llm.call(prompt)

@patch("os.environ", {"OPENAI_API_KEY": "key"})
def test_invalid_api_key(llm, prompt):
    with pytest.raises(AuthenticationError,match="Incorrect API key provided"):
        llm.call(prompt)

@patch("os.environ", {"OPENAI_API_KEY": "key"})
def test_successful_completion(llm,prompt):

    # Mock the litellm.completion function
    with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion') as completion_patch:
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
        assert kwargs["messages"] == [{"content": "Hello, how are you?", "role": "user"}]
        assert kwargs["model"] == "gpt-3.5-turbo"

@patch("os.environ", {"OPENAI_API_KEY": "key"})
def test_completion_with_extra_params(prompt):
    # Create an instance of LiteLLM
    llm = LiteLLM(model="gpt-3.5-turbo",extra_param=10)

    # Mock the litellm.completion function
    with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion') as completion_patch:

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

        assert kwargs["optional_params"]["extra_body"]["extra_param"] == 10