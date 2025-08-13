import traceback

from pandasai.agent.state import AgentState
from pandasai.core.prompts.base import BasePrompt

from .code_cleaning import CodeCleaner
from .code_validation import CodeRequirementValidator


class CodeGenerator:
    def __init__(self, context: AgentState):
        self._context = context
        self._code_cleaner = CodeCleaner(self._context)
        self._code_validator = CodeRequirementValidator(self._context)

    def generate_code(self, prompt: BasePrompt) -> str:
        """
        Generates code using a given LLM and performs validation and cleaning steps.

        Args:
            prompt (BasePrompt): The prompt to guide code generation.

        Returns:
            str: The final cleaned and validated code.

        Raises:
            Exception: If any step fails during the process.
        """
        try:
            self._context.logger.log(f"Using Prompt: {prompt}")

            # Generate the code
            code = self._context.config.llm.generate_code(prompt, self._context)
            # Store the original generated code (for logging purposes)
            self._context.last_code_generated = code
            self._context.logger.log(f"Code Generated:\n{code}")

            # Validate and clean the code
            cleaned_code = self.validate_and_clean_code(code)
            # Update with the final cleaned code (for subsequent processing and multi-turn conversations)
            self._context.last_code_generated = cleaned_code

            return cleaned_code

        except Exception as e:
            error_message = f"An error occurred during code generation: {e}"
            stack_trace = traceback.format_exc()

            self._context.logger.log(error_message)
            self._context.logger.log(f"Stack Trace:\n{stack_trace}")

            raise e

    def validate_and_clean_code(self, code: str) -> str:
        # Validate code requirements
        self._context.logger.log("Validating code requirements...")
        if not self._code_validator.validate(code):
            raise ValueError("Code validation failed due to unmet requirements.")
        self._context.logger.log("Code validation successful.")

        # Clean the code
        self._context.logger.log("Cleaning the generated code...")
        return self._code_cleaner.clean_code(code)
