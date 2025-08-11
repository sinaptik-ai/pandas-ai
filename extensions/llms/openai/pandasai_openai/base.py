from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union

from pandasai.core.prompts.base import BasePrompt
from pandasai.helpers.memory import Memory
from pandasai.llm.base import LLM

if TYPE_CHECKING:
    from pandasai.agent.state import AgentState


class BaseOpenAI(LLM):
    """Base class to implement a new OpenAI LLM.

    LLM base class, this class is extended to be used with OpenAI API.

    """

    api_token: str
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0
    max_tokens: int = 1000
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0.6
    best_of: int = 1
    n: int = 1
    stop: Optional[str] = None
    request_timeout: Union[float, Tuple[float, float], Any, None] = None
    max_retries: int = 2
    seed: Optional[int] = None
    # support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    client: Any
    _is_chat_model: bool

    def _set_params(self, **kwargs):
        """
        Set Parameters
        Args:
            **kwargs: ["model", "deployment_name", "temperature","max_tokens",
            "top_p", "frequency_penalty", "presence_penalty", "stop", "seed"]

        Returns:
            None.

        """

        valid_params = [
            "model",
            "deployment_name",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "stop": self.stop,
            "n": self.n,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        # Azure gpt-35-turbo doesn't support best_of
        # don't specify best_of if it is 1
        if self.best_of > 1:
            params["best_of"] = self.best_of

        return params

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        openai_creds: Dict[str, Any] = {}

        return {**openai_creds, **self._default_params}

    @property
    def _client_params(self) -> Dict[str, any]:
        return {
            "api_key": self.api_token,
            "base_url": self.api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
            "http_client": self.http_client,
        }

    def completion(self, prompt: str, memory: Memory) -> str:
        """
        Query the completion API

        Args:
            prompt (str): A string representation of the prompt.
            memory (Memory): Memory object containing conversation history.

        Returns:
            str: LLM response.

        """
        prompt = self.prepend_system_prompt(prompt, memory)

        params = {**self._invocation_params, "prompt": prompt}

        if self.stop is not None:
            params["stop"] = [self.stop]

        response = self.client.create(**params)

        self.last_prompt = prompt

        return response.choices[0].text

    def chat_completion(self, value: str, memory: Memory) -> str:
        """
        Query the chat completion API

        Args:
            value (str): Prompt
            memory (Memory): Memory object containing conversation history.

        Returns:
            str: LLM response.

        """
        messages = memory.to_openai_messages() if memory else []

        # adding current prompt as latest query message
        messages.append(
            {
                "role": "user",
                "content": value,
            },
        )

        params = {
            **self._invocation_params,
            "messages": messages,
        }

        if self.stop is not None:
            params["stop"] = [self.stop]

        response = self.client.create(**params)

        return response.choices[0].message.content

    def call(self, instruction: BasePrompt, context: AgentState = None):
        """
        Call the OpenAI LLM.

        Args:
            instruction (BasePrompt): A prompt object with instruction for LLM.
            context (AgentState): context to pass.

        Raises:
            UnsupportedModelError: Unsupported model

        Returns:
            str: Response
        """
        self.last_prompt = instruction.to_string()

        memory = context.memory if context else None

        return (
            self.chat_completion(self.last_prompt, memory)
            if self._is_chat_model
            else self.completion(self.last_prompt, memory)
        )
