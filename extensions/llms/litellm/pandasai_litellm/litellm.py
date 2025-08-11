from litellm import completion

from pandasai.agent.state import AgentState
from pandasai.core.prompts.base import BasePrompt
from pandasai.llm.base import LLM
import logging


class LiteLLM(LLM):
    """A lightweight wrapper for interacting with a specified LLM model.

    This class provides an interface to generate text based on user instructions
    using the specified language model. It allows for customization through additional
    parameters passed during initialization.

    Args:
        model (str): The name of the language model to use.
        **kwargs: Additional parameters for the model's completion settings.

    Properties:
        type (str): Returns the type of the LLM, which is 'litellm'.

    Methods:
        call(instruction: BasePrompt, _: AgentState = None) -> str:
            Generates a response based on the provided instruction."""

    def __init__(self, model: str, **kwargs):
        """
        Initializes the wrapper with the model name and any additional parameters.

        Args:
            model (str): The name of the LLM model.
            **kwargs: Any additional parameters required for completion.
        """
        super().__init__(api_key=None)
        self.model = model
        self.params = kwargs
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    @property
    def type(self) -> str:
        """Get the type of the model.

        This property returns the string representation of the model's type,
        which is 'litellm'.

        Returns:
            str: The type of the model."""
        return f"litellm"

    def call(self, instruction: BasePrompt, context: AgentState = None) -> str:
        """Generates a completion response based on the provided instruction.

        This method converts the given instruction into a user prompt string and
        sends it to a model for processing. It returns the content of the first
        message from the model's response.

        Args:
            instruction (BasePrompt): The instruction to convert into a prompt.
            context (AgentState, optional): An optional state of the agent. Defaults to None.

        Returns:
            str: The content of the model's response to the user prompt."""

        memory = context.memory if context else None
        self.last_prompt = self.prepend_system_prompt(instruction.to_string(), memory)

        return (
            completion(
                model=self.model,
                messages=[{"content": self.last_prompt, "role": "user"}],
                **self.params,
            )
            .choices[0]
            .message.content
        )
