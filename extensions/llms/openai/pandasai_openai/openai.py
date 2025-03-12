import os
from typing import Any, Dict, Optional

import openai
import requests

from pandasai.exceptions import APIKeyNotFoundError, UnsupportedModelError
from pandasai.helpers import load_dotenv

from .base import BaseOpenAI

load_dotenv()


class OpenAI(BaseOpenAI):
    """OpenAI LLM using BaseOpenAI Class.

    An API call to OpenAI API is sent and response is recorded and returned.
    The default chat model is **gpt-3.5-turbo**.
    The list of supported Chat models includes ["gpt-4", "gpt-4-0613", "gpt-4-32k",
     "gpt-4-32k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
     "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-instruct"].
    The list of supported Completion models includes "gpt-3.5-turbo-instruct" and
     "text-davinci-003" (soon to be deprecated).
    """

    _supported_chat_models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
    ]
    _supported_completion_models = ["gpt-3.5-turbo-instruct"]

    model: str = "gpt-4o-mini"

    def __init__(
        self,
        api_token: Optional[str] = None,
        **kwargs,
    ):
        """
        __init__ method of OpenAI Class

        Args:
            api_token (str): API Token for OpenAI platform.
            **kwargs: Extended Parameters inferred from BaseOpenAI class

        """
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or None

        if not self.api_token:
            raise APIKeyNotFoundError("OpenAI API key is required")

        self.api_base = (
            kwargs.get("api_base") or os.getenv("OPENAI_API_BASE") or self.api_base
        )
        self.openai_proxy = kwargs.get("openai_proxy") or os.getenv("OPENAI_PROXY")
        if self.openai_proxy:
            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}

        self._set_params(**kwargs)
        # set the client
        model_name = self.model.split(":")[1] if "ft:" in self.model else self.model
        if model_name in self._supported_chat_models:
            self._is_chat_model = True
            self.client = openai.OpenAI(**self._client_params).chat.completions
        elif model_name in self._supported_completion_models:
            self._is_chat_model = False
            self.client = openai.OpenAI(**self._client_params).completions
        else:
            self._is_chat_model = kwargs.get("is_chat_model", True)
            model_names = [
                model.get("id")
                for model in requests.get(f"{self.api_base}/models")
                .json()
                .get("data", [])
            ]
            if self.model in model_names:
                self.client = (
                    openai.OpenAI(**self._client_params).chat.completions
                    if self._is_chat_model
                    else openai.OpenAI(**self._client_params).completions
                )
            else:
                raise UnsupportedModelError(self.model)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API"""
        return {
            **super()._default_params,
            "model": self.model,
        }

    @property
    def type(self) -> str:
        return "openai"
