from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pandasai.core.prompts.base import BasePrompt
from pandasai.helpers import load_dotenv
from pandasai.llm.base import LLM

if TYPE_CHECKING:
    from pandasai.agent.state import AgentState


load_dotenv()


class HuggingFaceTextGen(LLM):
    """HuggingFace Text Generation Inference LLM
       Generates text using HuggingFace inference API.

    Attributes:
        max_new_tokens: Max number of tokens to generate.
        top_k: Sample from top k tokens.
        top_p: Sample from top p probability.
        typical_p: typical probability of a token.
        temperature: Controls randomness in the model. Lower values will make the model more deterministic and higher values will make the model more random.
        repetition_penalty: controls the likelihood of repeating same tokens based on value.
        truncate: truncate the input to the maximum length of the model.
        stop_sequences: A stop sequence is a string that stops the model from generating tokens.
        seed: The seed to use for random generation.
        do_sample: Whether or not to use sampling.
        timeout: adding timeout restricts huggingface from waiting indefinitely for model's response.
    """

    max_new_tokens: int = 1024
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.8
    typical_p: Optional[float] = 0.8
    temperature: float = 1e-3  # must be strictly positive
    repetition_penalty: Optional[float] = None
    truncate: Optional[int] = None
    stop_sequences: List[str] = []
    seed: Optional[int] = None
    do_sample: Optional[bool] = False
    inference_server_url: str = ""
    streaming: Optional[bool] = False
    timeout: int = 120
    client: Any

    def __init__(self, inference_server_url: str, **kwargs):
        """Initializes an instance with a connection to a text generation inference server.

    This constructor sets up a client to communicate with a specified text generation 
    inference server. Additional configuration can be passed via keyword arguments 
    which will be set as attributes if they match the class annotations.

    Args:
        inference_server_url (str): The URL of the inference server for text generation.
        **kwargs: Additional optional configuration settings as keyword arguments.

    Raises:
        ImportError: If the `text_generation` package is not installed."""
        try:
            import text_generation

            for key, val in kwargs.items():
                if key in self.__annotations__:
                    setattr(self, key, val)

            self.client = text_generation.Client(
                base_url=inference_server_url,
                timeout=self.timeout,
            )

        except ImportError as e:
            raise ImportError(
                "Could not import text_generation python package. "
                "Please install it with `pip install text_generation`."
            ) from e

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling text generation inference API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "truncate": self.truncate,
            "stop_sequences": self.stop_sequences,
            "do_sample": self.do_sample,
            "seed": self.seed,
        }

    def call(self, instruction: BasePrompt, context: AgentState = None) -> str:
        """Generates a text response based on a given instruction and context.

    This method converts the instruction into a string and optionally
    incorporates memory from the provided context to form a complete prompt.
    It then generates a text response using a client, with an option for
    streaming the response. If stop sequences are defined, they are removed
    from the end of the generated text.

    Args:
        instruction (BasePrompt): The instruction containing the prompt
            to be converted into a string for text generation.
        context (AgentState, optional): The context providing additional
            memory data to be included in the prompt, if available.

    Returns:
        str: The generated text response, potentially modified to exclude
        specified stop sequences."""
        prompt = instruction.to_string()

        memory = context.memory if context else None

        prompt = self.prepend_system_prompt(prompt, memory)

        params = self._default_params
        if self.streaming:
            return "".join(
                chunk.template
                for chunk in self.client.generate_stream(prompt, **params)
            )
        res = self.client.generate(prompt, **params)
        if self.stop_sequences:
            # remove stop sequences from the end of the generated text
            for stop_seq in self.stop_sequences:
                if stop_seq in res.generated_text:
                    res.generated_text = res.generated_text[
                        : res.generated_text.index(stop_seq)
                    ]
        self.last_prompt = prompt
        return res.generated_text

    @property
    def type(self) -> str:
        """Gets the type identifier for the text generation model.

    This property returns a string that specifies the type of model being used 
    for text generation, which is 'huggingface-text-generation'.

    Returns:
        str: The type identifier 'huggingface-text-generation'."""
        return "huggingface-text-generation"
