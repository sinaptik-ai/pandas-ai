import ast


class Sandbox:
    def __init__(self):
        self._started: bool = False

    def start(self):
        raise NotImplementedError("The start method must be implemented by subclasses.")

    def stop(self):
        raise NotImplementedError("The stop method must be implemented by subclasses.")

    def execute(self, code: str, environment: dict) -> dict:
        if not self._started:
            self.start()
            return self._exec_code(code, environment)

        return self._exec_code(code, environment)

    def _exec_code(self, code: str, environment: dict) -> dict:
        raise NotImplementedError("Subclasses must implement the _exec_code method.")

    def transfer_file(self, csv_data, filename="file.csv"):
        raise NotImplementedError(
            "The transfer_file method must be implemented by subclasses."
        )

    def _extract_sql_queries_from_code(self, code) -> list[str]:
        """
        Extract SQL query strings from Python code

        Args:
            code (str): Python code as a string.

        Returns:
            list: List of SQL query strings found in the code.
        """
        sql_queries = []

        class SQLQueryExtractor(ast.NodeVisitor):
            def visit_Assign(self, node):
                # Look for assignments where SQL queries might be defined
                if (
                    isinstance(node.value, (ast.Str, ast.Constant))
                    and isinstance(node.value.s, str)
                    and any(
                        keyword in node.value.s.upper()
                        for keyword in ["SELECT", "WITH"]
                    )
                ):
                    sql_queries.append(node.value.s)
                self.generic_visit(node)

            def visit_Call(self, node):
                # Look for function calls where SQL queries might be passed
                for arg in node.args:
                    if (
                        isinstance(arg, (ast.Str, ast.Constant))
                        and isinstance(arg.s, str)
                        and any(
                            keyword in arg.s.upper() for keyword in ["SELECT", "WITH"]
                        )
                    ):
                        sql_queries.append(arg.s)
                self.generic_visit(node)

        # Parse the code into an AST and visit all nodes
        tree = ast.parse(code)
        SQLQueryExtractor().visit(tree)

        return sql_queries

    def _compile_code(self, code: str) -> str:
        """Compile code as a Python module

        Args:
            code (str): Code as a string to compile.

        Raises:
            SyntaxError: If the code contains syntax errors.

        Returns:
            str: Compiled code as a string.
        """
        try:
            return compile(code, "<string>", "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in code: {e}") from e
