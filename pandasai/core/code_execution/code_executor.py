from typing import Any

from pandasai.config import Config
from pandasai.core.code_execution.environment import get_environment
from pandasai.exceptions import CodeExecutionError, NoResultFoundError


class CodeExecutor:
    """
    Handle the logic on how to handle different lines of code
    """

    _environment: dict

    def __init__(self, config: Config) -> None:
        self._environment = get_environment()

    def add_to_env(self, key: str, value: Any) -> None:
        """
        Expose extra variables in the code to be used
        Args:
            key (str): Name of variable or lib alias
            value (Any): It can any value int, float, function, class etc.
        """
        self._environment[key] = value

    def execute(
        self, code: str, context: Optional[Dict[str, Any]] = None
    ) -> CodeExecutionResult:
        """
        Execute the code and return the result.

        Args:
            code (str): The code to execute.
            context (Dict[str, Any], optional): The context to execute the code in.

        Returns:
            CodeExecutionResult: The result of the code execution.
        """
