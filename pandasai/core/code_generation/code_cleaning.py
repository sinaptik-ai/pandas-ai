import ast
import os.path
import re
import uuid
from pathlib import Path

import astor

from pandasai.agent.state import AgentState
from pandasai.constants import DEFAULT_CHART_DIRECTORY
from pandasai.core.code_execution.code_executor import CodeExecutor
from pandasai.query_builders.sql_parser import SQLParser

from ...exceptions import MaliciousQueryError


class CodeCleaner:
    def __init__(self, context: AgentState):
        """
        Initialize the CodeCleaner with the provided context.

        Args:
            context (AgentState): The pipeline context for cleaning and validation.
        """
        self.context = context

    def _check_direct_sql_func_def_exists(self, node: ast.AST) -> bool:
        """
        Check if the node defines a direct SQL execution function.
        """
        return isinstance(node, ast.FunctionDef) and node.name == "execute_sql_query"

    def _replace_table_names(
        self, sql_query: str, table_names: list, allowed_table_names: dict
    ) -> str:
        """
        Replace table names in the SQL query with case-sensitive or authorized table names.
        """
        regex_patterns = {
            table_name: re.compile(r"\b" + re.escape(table_name) + r"\b")
            for table_name in table_names
        }
        for table_name in table_names:
            if table_name in allowed_table_names:
                quoted_table_name = allowed_table_names[table_name]
                sql_query = regex_patterns[table_name].sub(quoted_table_name, sql_query)
            else:
                raise MaliciousQueryError(
                    f"Query uses unauthorized table: {table_name}."
                )
        return sql_query

    def _clean_sql_query(self, sql_query: str) -> str:
        """
        Clean the SQL query by trimming semicolons and validating table names.
        """
        sql_query = sql_query.rstrip(";")
        table_names = SQLParser.extract_table_names(sql_query)
        allowed_table_names = {
            df.schema.name: df.schema.name for df in self.context.dfs
        } | {f'"{df.schema.name}"': df.schema.name for df in self.context.dfs}

        return self._replace_table_names(sql_query, table_names, allowed_table_names)

    def _validate_and_make_table_name_case_sensitive(self, node: ast.AST) -> ast.AST:
        """
        Validate table names and convert them to case-sensitive names in the SQL query.
        """
        if isinstance(node, ast.Assign):
            if (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ["sql_query", "query"]
            ):
                sql_query = self._clean_sql_query(node.value.value)
                node.value.value = sql_query
            elif (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "execute_sql_query"
                and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant)
                and isinstance(node.value.args[0].value, str)
            ):
                sql_query = self._clean_sql_query(node.value.args[0].value)
                node.value.args[0].value = sql_query

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if (
                isinstance(node.value.func, ast.Name)
                and node.value.func.id == "execute_sql_query"
                and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant)
                and isinstance(node.value.args[0].value, str)
            ):
                sql_query = self._clean_sql_query(node.value.args[0].value)
                node.value.args[0].value = sql_query

        return node

    def extract_fix_dataframe_redeclarations(
        self,
        node: ast.AST,
        code_lines: list[str],
    ) -> ast.AST:
        """
        Checks if dataframe reclaration in the code like pd.DataFrame({...})
        Args:
            node (ast.AST): Code Node
            code_lines (list[str]): List of code str line by line

        Returns:
            ast.AST: Updated Ast Node fixing redeclaration
        """
        if isinstance(node, ast.Assign):
            target_names, is_slice, target = self.get_target_names(node.targets)

            if target_names and self.check_is_df_declaration(node):
                # Construct dataframe from node
                code = "\n".join(code_lines)
                code_executor = CodeExecutor(self.context.config)
                env = code_executor.execute(code)

                df_generated = (
                    env[target_names[0]][target.slice.value]
                    if is_slice
                    else env[target_names[0]]
                )

                # check if exists in provided dfs
                for index, df in enumerate(self.context.dfs):
                    head = df.get_head()
                    if head.shape == df_generated.shape and head.columns.equals(
                        df_generated.columns
                    ):
                        target_var = (
                            ast.Subscript(
                                value=ast.Name(id=target_names[0], ctx=ast.Load()),
                                slice=target.slice,
                                ctx=ast.Store(),
                            )
                            if is_slice
                            else ast.Name(id=target_names[0], ctx=ast.Store())
                        )
                        return ast.Assign(
                            targets=[target_var],
                            value=ast.Subscript(
                                value=ast.Name(id="dfs", ctx=ast.Load()),
                                slice=ast.Index(value=ast.Num(n=index)),
                                ctx=ast.Load(),
                            ),
                        )
        return None

    def get_target_names(self, targets):
        target_names = []
        is_slice = False

        for target in targets:
            if isinstance(target, ast.Name) or (
                isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            ):
                target_names.append(
                    target.id if isinstance(target, ast.Name) else target.value.id
                )
                is_slice = isinstance(target, ast.Subscript)

        return target_names, is_slice, target

    def check_is_df_declaration(self, node: ast.AST):
        value = node.value
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and hasattr(value.func.value, "id")
            and value.func.value.id == "pd"
            and value.func.attr == "DataFrame"
        )

    def clean_code(self, code: str) -> str:
        """
        Clean the provided code by validating imports, handling SQL queries, and processing charts.

        Args:
            code (str): The code to clean.

        Returns:
            tuple: Cleaned code as a string and a list of additional dependencies.
        """
        code = self._replace_output_filenames_with_temp_chart(code)

        # If plt.show is in the code, remove that line
        code = re.sub(r"plt.show\(\)", "", code)

        clean_code_lines = []

        tree = ast.parse(code)
        new_body = []

        for node in tree.body:
            if self._check_direct_sql_func_def_exists(node):
                continue

            node = self._validate_and_make_table_name_case_sensitive(node)

            clean_code_lines.append(astor.to_source(node))

            new_body.append(
                self.extract_fix_dataframe_redeclarations(node, clean_code_lines)
                or node
            )

        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree, pretty_source=lambda x: "".join(x)).strip()

    def _replace_output_filenames_with_temp_chart(self, code: str) -> str:
        """
        Replace output file names with "temp_chart.png".
        """
        _id = uuid.uuid4()
        chart_path = os.path.join(DEFAULT_CHART_DIRECTORY, f"temp_chart_{_id}.png")
        return re.sub(
            r"""(['"])([^'"]*\.png)\1""",
            lambda m: f"{m.group(1)}{chart_path}{m.group(1)}",
            code,
        )
