import os
from typing import Any, Dict, Optional

import openai
from pandasai.llm.base import LLM
from pandasai.core.prompts.base import BasePrompt
from pandasai.agent.state import AgentState
from pandasai.helpers import load_dotenv
from pandasai.core.prompts.generate_system_message import GenerateSystemMessagePrompt

# Load .env if present
load_dotenv()

class OllamaLLM(LLM):
    _type: str = "ollama"
    model: str = "llama2"  # default model, can be overridden

    def __init__(
        self,
        api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # For Ollama, a dummy API key is required (but not used)
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        # Get base URL from parameter or env, default to localhost
        base = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Ensure the base URL ends with "/v1"
        if not base.rstrip("/").endswith("/v1"):
            base = base.rstrip("/") + "/v1"
        self.base_url = base

        # Set additional parameters (e.g. model, temperature, etc.)
        self._set_params(**kwargs)

        # Assume chat mode by default
        self._is_chat_model = True
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self._client_params,
        ).chat.completions

    def _set_params(self, **kwargs: Any) -> None:
        valid_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "best_of",
            "request_timeout",
            "max_retries",
            "seed",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    @property
    def _client_params(self) -> Dict[str, Any]:
        return {
            "timeout": getattr(self, "request_timeout", None),
            "max_retries": getattr(self, "max_retries", 2),
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": getattr(self, "temperature", 0),
            "top_p": getattr(self, "top_p", 1),
            "frequency_penalty": getattr(self, "frequency_penalty", 0),
            "presence_penalty": getattr(self, "presence_penalty", 0),
            "n": getattr(self, "n", 1),
        }
        if hasattr(self, "max_tokens") and self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if hasattr(self, "stop") and self.stop is not None:
            params["stop"] = [self.stop]
        if hasattr(self, "best_of") and self.best_of > 1:
            params["best_of"] = self.best_of
        return params

    def call(self, instruction: BasePrompt, context: AgentState = None) -> str:
        # Get the base prompt string from the user instruction.
        prompt_str = instruction.to_string()
        # If a context is provided with conversation memory,
        # prepend the system prompt (generated via GenerateSystemMessagePrompt).
        if context and context.memory:
            system_prompt = GenerateSystemMessagePrompt(memory=context.memory)
            prompt_str = system_prompt.to_string() + "\n" + prompt_str

        if self._is_chat_model:
            response = self.client.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_str}],
                **self._default_params,
            )
            return response.choices[0].message.content
        else:
            response = self.client.create(
                model=self.model,
                prompt=prompt_str,
                **self._default_params,
            )
            return response.choices[0].text

    @property
    def type(self) -> str:
        return self._type
