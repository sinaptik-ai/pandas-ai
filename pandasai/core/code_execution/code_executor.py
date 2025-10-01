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

    def execute(self, code: str) -> dict:
        try:
            exec(code, self._environment)
        except Exception as e:
            raise CodeExecutionError("Code execution failed") from e
        return self._environment

    def execute_and_return_result(self, code: str) -> Any:
        """
        Executes the return updated environment
        """
        self.execute(code)

        # Get the result
        if "result" not in self._environment:
            raise NoResultFoundError(
                "No result was returned from the code execution. Please return the result in dictionary format, for example: result = {'type': ..., 'value': ...}"
            )

        return self._environment.get("result", None)

    @property
    def environment(self) -> dict:
        return self._environment
