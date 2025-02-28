from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

from pandasai.core.prompts.base import BasePrompt
from pandasai.exceptions import APIKeyNotFoundError, UnsupportedModelError
from pandasai.helpers import load_dotenv
from pandasai.llm.base import LLM

if TYPE_CHECKING:
    from pandasai.agent.state import AgentState


load_dotenv()


class BedrockClaude(LLM):
    """Bedrock Claude LLM
       Generates text using Anthropic Claude Messages API.

    Attributes:
        bedrock_runtime_client: The boto3 bedrock runtime client.
        max_tokens: Max number of tokens to generate.
        model: The Bedrock Claude model to use.
        temperature: (Optional) The amount of randomness injected into the response.
        top_p: (Optional) Use nucleus sampling. In nucleus sampling, Anthropic Claude computes the cumulative distribution over all the options for each subsequent token in decreasing probability order and cuts it off once it reaches a particular probability specified by top_p. You should alter either temperature or top_p, but not both.
        top_k: (Optional) Only sample from the top K options for each subsequent token.
        stop_sequences: (Optional) Custom text sequences that cause the model to stop generating. Anthropic Claude models normally stop when they have naturally completed their turn, in this case the value of the stop_reason response field is end_turn. If you want the model to stop generating when it encounters custom strings of text, you can use the stop_sequences parameter. If the model encounters one of the custom text strings, the value of the stop_reason response field is stop_sequence and the value of stop_sequence contains the matched stop sequence.
    """

    _supported__models = [
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
    ]
    _valid_params = [
        "max_tokens",
        "model",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
    ]

    max_tokens: int = 1024
    model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stop_sequences: Optional[str] = None
    client: Any

    def __init__(self, bedrock_runtime_client, **kwargs):
        for key, val in kwargs.items():
            if key in self._valid_params:
                setattr(self, key, val)

        self.client = bedrock_runtime_client

        if self.model not in self._supported__models:
            raise UnsupportedModelError(self.model)

        invoke_model = getattr(self.client, "invoke_model", None)
        if not callable(invoke_model):
            raise APIKeyNotFoundError

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling text generation inference API."""
        return {
            "max_tokens": self.max_tokens,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences,
        }

    def call(self, instruction: BasePrompt, context: AgentState = None) -> str:
        prompt = instruction.to_string()
        memory = context.memory if context else None

        messages = []
        system_prompt = ""
        if memory:
            if memory.agent_description:
                system_prompt = memory.agent_description

            for message in memory.all():
                if message["is_user"]:
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"].append(
                            {"type": "text", "text": message["message"]}
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": message["message"]}
                                ],
                            }
                        )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": message["message"]}],
                        }
                    )

        # adding current prompt as latest query message
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"].append({"type": "text", "text": prompt})
        else:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            )

        params = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": messages,
        }
        for key, value in self._default_params.items():
            if key != "model" and value is not None:
                params[key] = value

        body = json.dumps(params)
        # print(body)

        response = self.client.invoke_model(modelId=self.model, body=body)

        response_body = json.loads(response.get("body").read())

        self.last_prompt = prompt
        # print(response_body["content"][0]["text"])
        return response_body["content"][0]["text"]

    @property
    def type(self) -> str:
        return "bedrock-claude"
