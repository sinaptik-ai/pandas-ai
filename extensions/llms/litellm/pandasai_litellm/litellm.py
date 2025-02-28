from litellm import completion

from pandasai.agent.state import AgentState
from pandasai.core.prompts.base import BasePrompt
from pandasai.llm.base import LLM
import logging

class LiteLLM(LLM):
    """Base class to implement a new OpenAI LLM.

    LLM base class, this class is extended to be used with OpenAI API.

    """

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
        return f"litellm"

    def call(self, instruction: BasePrompt, _: AgentState = None) -> str:

        user_prompt = instruction.to_string()

        return completion(
            model=self.model,
            messages=[{"content":user_prompt,"role":"user"}],
            **self.params
        ).choices[0].message.content
