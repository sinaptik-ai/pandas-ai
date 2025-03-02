import os
from types import SimpleNamespace
from unittest import mock

import pytest

from extensions.llms.ollama.pandasai_ollama.ollama import OllamaLLM
from pandasai.core.prompts.base import BasePrompt


# Helper to simulate an OpenAI-like response.
def create_response(content, mode="chat"):
    if mode == "chat":
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
    else:
        return SimpleNamespace(choices=[SimpleNamespace(text=content)])


# Dummy prompt for testing.
class DummyPrompt(BasePrompt):
    template = "Tell me a joke."

    def __init__(self, message):
        super().__init__(message=message)
        self.props = {"message": message}

    def to_string(self):
        return self.props.get("message", "")


@pytest.fixture
def prompt():
    return DummyPrompt("Hello")


class TestOllamaLLM:
    def test_default_base_url(self, prompt):
        # Remove OLLAMA_BASE_URL from environment, if exists.
        os.environ.pop("OLLAMA_BASE_URL", None)

        with mock.patch(
            "extensions.llms.ollama.pandasai_ollama.ollama.openai.OpenAI"
        ) as mock_openai:
            dummy_client = mock.Mock()
            dummy_completion = mock.Mock()
            expected_response = "Ollama dummy response"
            dummy_completion.create.return_value = create_response(
                expected_response, mode="chat"
            )
            dummy_client.chat.completions = dummy_completion
            mock_openai.return_value = dummy_client

            llm = OllamaLLM(api_key="dummy")
            # Default base URL should be "http://localhost:11434/v1"
            assert llm.base_url == "http://localhost:11434/v1"

            result = llm.call(prompt)
            dummy_completion.create.assert_called_once_with(
                model="llama2",
                messages=[{"role": "user", "content": prompt.to_string()}],
                **llm._default_params,
            )
            assert result == expected_response

    def test_overridden_base_url(self, prompt):
        custom_url = "http://custom-url:1234"
        with mock.patch(
            "extensions.llms.ollama.pandasai_ollama.ollama.openai.OpenAI"
        ) as mock_openai:
            dummy_client = mock.Mock()
            dummy_completion = mock.Mock()
            expected_response = "Custom joke"
            dummy_completion.create.return_value = create_response(
                expected_response, mode="chat"
            )
            dummy_client.chat.completions = dummy_completion
            mock_openai.return_value = dummy_client

            llm = OllamaLLM(api_key="dummy", ollama_base_url=custom_url)
            expected_url = custom_url.rstrip("/") + "/v1"
            assert llm.base_url == expected_url

            result = llm.call(prompt)
            dummy_completion.create.assert_called_once_with(
                model="llama2",
                messages=[{"role": "user", "content": prompt.to_string()}],
                **llm._default_params,
            )
            assert result == expected_response

    def test_params_setting(self, prompt):
        with mock.patch(
            "extensions.llms.ollama.pandasai_ollama.ollama.openai.OpenAI"
        ) as mock_openai:
            dummy_client = mock.Mock()
            dummy_completion = mock.Mock()
            dummy_completion.create.return_value = create_response(
                "Dummy response", mode="chat"
            )
            dummy_client.chat.completions = dummy_completion
            mock_openai.return_value = dummy_client

            llm = OllamaLLM(
                api_key="dummy", model="llama2", temperature=0.7, max_tokens=150
            )
            assert llm.model == "llama2"
            assert llm.temperature == 0.7
            assert llm.max_tokens == 150

    def test_call_non_chat_mode(self, prompt):
        # Force non-chat mode.
        with mock.patch(
            "extensions.llms.ollama.pandasai_ollama.ollama.openai.OpenAI"
        ) as mock_openai:
            dummy_client = mock.Mock()
            dummy_completion = mock.Mock()
            expected_response = "Non-chat dummy response"
            dummy_completion.create.return_value = create_response(
                expected_response, mode="non-chat"
            )
            # In non-chat mode, the client uses the 'completions' attribute.
            dummy_client.completions = dummy_completion
            mock_openai.return_value = dummy_client

            llm = OllamaLLM(api_key="dummy", model="llama2")
            llm._is_chat_model = False  # Force non-chat mode.
            # Set client to completions for non-chat mode.
            llm.client = dummy_client.completions
            result = llm.call(prompt)
            dummy_completion.create.assert_called_once_with(
                model="llama2",
                prompt=prompt.to_string(),
                **llm._default_params,
            )
            assert result == expected_response

    def test_type_property(self):
        with mock.patch(
            "extensions.llms.ollama.pandasai_ollama.ollama.openai.OpenAI"
        ) as mock_openai:
            dummy_client = mock.Mock()
            dummy_completion = mock.Mock()
            dummy_completion.create.return_value = create_response(
                "Dummy response", mode="chat"
            )
            dummy_client.chat.completions = dummy_completion
            mock_openai.return_value = dummy_client

            llm = OllamaLLM(api_key="dummy")
            assert llm.type == "ollama"
