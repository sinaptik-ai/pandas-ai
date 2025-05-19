"""Test BambooLLM class."""

import unittest
from unittest.mock import MagicMock, patch

from pandasai.core.prompts.base import BasePrompt
from pandasai.exceptions import PandasAIApiCallError
from pandasai.llm.bamboo_llm import BambooLLM


class MockBasePrompt(BasePrompt):
    template: str = "instruction"

    def to_json(self):
        return {
            "code": ["print('Hello')", "for i in range(10): print(i)"],
            "query": ["What is Chroma?", "How does it work?"],
        }


class TestBambooLLM(unittest.TestCase):
    def get_prompt(self):
        return MockBasePrompt()

    def get_context(self):
        return MagicMock()

    @patch("pandasai.helpers.session.Session.post")
    def test_call_method(self, mock_post):
        prompt = self.get_prompt()
        context = self.get_context()
        mock_post.return_value = {"answer": "Hello World"}
        bllm = BambooLLM(api_key="dummy_key")
        bllm.call(prompt, context)
        mock_post.assert_called_once_with(
            "/query",
            json={"prompt": "instruction"},
        )

    @patch("pandasai.helpers.session.Session.post")
    def test_status_code_200(self, mock_post):
        prompt = self.get_prompt()
        context = self.get_context()
        mock_post.return_value = {"answer": "Hello World"}
        bllm = BambooLLM(api_key="dummy_key")
        bllm.call(prompt, context)

    @patch("pandasai.helpers.session.Session.post")
    def test_status_code_not_200(self, mock_post):
        prompt = self.get_prompt()
        context = self.get_context()
        mock_post.side_effect = PandasAIApiCallError("Invalid API key")
        bllm = BambooLLM(api_key="dummy_key")
        with self.assertRaises(PandasAIApiCallError):
            bllm.call(prompt, context)
