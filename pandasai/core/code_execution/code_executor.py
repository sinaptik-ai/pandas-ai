"""Code Executor module."""

import ast
import sys
from typing import Any, Dict, List, Optional, Union

from pandasai.core.code_execution.code_execution_context import CodeExecutionContext
from pandasai.core.code_execution.code_execution_output import CodeExecutionOutput
from pandasai.core.code_execution.code_execution_result import CodeExecutionResult
from pandasai.core.code_execution.code_execution_status import CodeExecutionStatus
from pandasai.core.code_execution.code_execution_type import CodeExecutionType
from pandasai.core.code_execution.code_execution_warning import CodeExecutionWarning
from pandasai.core.code_execution.code_execution_error import CodeExecutionError


class CodeExecutor:
    """Code Executor class."""

    def execute(
        self, code: str, context: CodeExecutionContext
    ) -> CodeExecutionResult:
        """
        Execute the code and return the result.

        Args:
            code (str): The code to execute
            context (CodeExecutionContext): The context to execute the code in

        Returns:
            CodeExecutionResult: The result of the code execution
        """
        # Create a restricted globals dictionary for safer execution
        restricted_globals = {
            "__builtins__": {
                name: getattr(__builtins__, name)
                for name in dir(__builtins__)
                if name not in ["eval", "exec", "compile", "__import__"]
            }
        }
        
        # Add context variables to globals
        for key, value in context.variables.items():
            restricted_globals[key] = value

        # Create a local namespace for execution
        local_namespace = {}
        
        try:
            # Parse the code to validate it before execution
            ast.parse(code)
            
            # Execute the code in the restricted environment
            compiled_code = compile(code, "<string>", "exec")
            exec(compiled_code, restricted_globals, local_namespace)
            
            return CodeExecutionResult(
                status=CodeExecutionStatus.SUCCESS,
                output=CodeExecutionOutput(
                    type=CodeExecutionType.PYTHON_OBJECT,
                    value=local_namespace.get("result", None),
                ),
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            
            return CodeExecutionResult(
                status=CodeExecutionStatus.ERROR,
                error=CodeExecutionError(
                    type=error_type,
                    message=error_message,
                ),
            )
