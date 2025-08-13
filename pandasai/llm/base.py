from __future__ import annotations

import ast
import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from pandasai.core.prompts.base import BasePrompt
from pandasai.core.prompts.generate_system_message import GenerateSystemMessagePrompt
from pandasai.helpers.memory import Memory

from ..exceptions import (
    APIKeyNotFoundError,
    MethodNotImplementedError,
    NoCodeFoundError,
)

if TYPE_CHECKING:
    from pandasai.agent.state import AgentState


class LLM:
    """Base class to implement a new LLM."""

    last_prompt: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize LLM.

        Args:
            api_key (Optional[str], optional): API key for LLM. Defaults to None.
            **kwargs (Any): Additional arguments.
        """
        self.api_key = api_key

    def is_pandasai_llm(self) -> bool:
        """
        Return True if the LLM is from pandasAI.

        Returns:
            bool: True if the LLM is from pandasAI

        """
        return True

    @property
    def type(self) -> str:
        """
        Return type of LLM.

        Raises:
            APIKeyNotFoundError: Type has not been implemented

        Returns:
            str: Type of LLM a string

        """
        raise APIKeyNotFoundError("Type has not been implemented")

    def _polish_code(self, code: str) -> str:
        """
        Polish the code by removing the leading "python" or "py",  \
        removing surrounding '`' characters  and removing trailing spaces and new lines.

        Args:
            code (str): A string of Python code.

        Returns:
            str: Polished code.

        """
        if re.match(r"^(python|py)", code):
            code = re.sub(r"^(python|py)", "", code)
        if re.match(r"^`.*`$", code):
            code = re.sub(r"^`(.*)`$", r"\1", code)
        code = code.strip()
        return code

    def _is_python_code(self, string):
        """
        Return True if it is valid python code.
        Args:
            string (str):

        Returns (bool): True if Python Code otherwise False

        """
        try:
            ast.parse(string)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, response: str, separator: str = "```") -> str:
        """
        Extract the code from the response.

        Args:
            response (str): Response
            separator (str, optional): Separator. Defaults to "```".

        Raises:
            NoCodeFoundError: No code found in the response

        Returns:
            str: Extracted code from the response

        """
        code = response

        # If separator is in the response then we want the code in between only
        if separator in response and len(code.split(separator)) > 1:
            code = code.split(separator)[1]
        code = self._polish_code(code)

        # Even if the separator is not in the response, the output might still be valid python code
        if not self._is_python_code(code):
            raise NoCodeFoundError("No code found in the response")

        return code

    def prepend_system_prompt(self, prompt: str, memory: Memory) -> str | Any:
        """
        Append system prompt to the chat prompt, useful when model doesn't have messages for chat history
        Args:
            prompt (str): prompt for chat method
            memory (Memory): user conversation history
        """
        return self.get_system_prompt(memory) + prompt if memory else prompt

    def get_system_prompt(self, memory: Memory) -> Any:
        """
        Generate system prompt with agent info and previous conversations
        """
        system_prompt = GenerateSystemMessagePrompt(memory=memory)
        return system_prompt.to_string()

    def get_messages(self, memory: Memory) -> Any:
        """
        Return formatted messages
        Args:
            memory (Memory): Get past Conversation from memory
        """
        return memory.get_previous_conversation()

    @abstractmethod
    def call(self, instruction: BasePrompt, context: AgentState = None) -> str:
        """
        Execute the LLM with given prompt.

        Args:
            instruction (BasePrompt): A prompt object with instruction for LLM.
            context (AgentState, optional): AgentState. Defaults to None.

        Raises:
            MethodNotImplementedError: Call method has not been implemented

        """
        raise MethodNotImplementedError("Call method has not been implemented")

    def generate_code(self, instruction: BasePrompt, context: AgentState) -> str:
        """
        Generate the code based on the instruction and the given prompt.

        Args:
            instruction (BasePrompt): Prompt with instruction for LLM.
            context (AgentState): Context to pass.

        Returns:
            str: A string of Python code.

        """
        response = self.call(instruction, context)
        return self._extract_code(response)
