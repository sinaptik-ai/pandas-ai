import ast
import os
import re
import unittest
from unittest.mock import MagicMock

from pandasai.agent.state import AgentState
from pandasai.core.code_generation.code_cleaning import CodeCleaner
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import MaliciousQueryError


class TestCodeCleaner(unittest.TestCase):
    def setUp(self):
        # Setup a mock context for CodeCleaner
        self.context = MagicMock(spec=AgentState)
        self.cleaner = CodeCleaner(self.context)
        self.sample_df = DataFrame(
            {
                "country": ["United States", "United Kingdom", "Japan", "China"],
                "gdp": [
                    19294482071552,
                    2891615567872,
                    4380756541440,
                    14631844184064,
                ],
                "happiness_index": [6.94, 7.22, 5.87, 5.12],
            }
        )

    def test_check_direct_sql_func_def_exists_true(self):
        node = ast.FunctionDef(
            name="execute_sql_query",
            args=ast.arguments(
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[],
            decorator_list=[],
            returns=None,
        )
        result = self.cleaner._check_direct_sql_func_def_exists(node)
        self.assertTrue(result)

    def test_replace_table_names_valid(self):
        sql_query = "SELECT * FROM my_table;"
        table_names = ["my_table"]
        allowed_table_names = {"my_table": "my_table"}
        result = self.cleaner._replace_table_names(
            sql_query, table_names, allowed_table_names
        )
        self.assertEqual(result, "SELECT * FROM my_table;")

    def test_replace_table_names_invalid(self):
        sql_query = "SELECT * FROM my_table;"
        table_names = ["my_table"]
        allowed_table_names = {}
        with self.assertRaises(MaliciousQueryError):
            self.cleaner._replace_table_names(
                sql_query, table_names, allowed_table_names
            )

    def test_clean_sql_query(self):
        sql_query = "SELECT * FROM my_table;"
        mock_dataframe = MagicMock(spec=object)
        mock_dataframe.name = "my_table"
        mock_dataframe.schema = MagicMock()
        mock_dataframe.schema.name = "my_table"
        self.cleaner.context.dfs = [mock_dataframe]
        result = self.cleaner._clean_sql_query(sql_query)
        self.assertEqual(result, "SELECT * FROM my_table")

    def test_validate_and_make_table_name_case_sensitive(self):
        node = ast.Assign(
            targets=[ast.Name(id="query", ctx=ast.Store())],
            value=ast.Constant(value="SELECT * FROM my_table"),
        )
        mock_dataframe = MagicMock(spec=object)
        mock_dataframe.name = "my_table"
        self.cleaner.context.dfs = [mock_dataframe]
        mock_dataframe.schema = MagicMock()
        mock_dataframe.schema.name = "my_table"
        updated_node = self.cleaner._validate_and_make_table_name_case_sensitive(node)
        self.assertEqual(updated_node.value.value, "SELECT * FROM my_table")

    def test_replace_output_filenames_with_temp_chart(self):
        handler = self.cleaner
        handler.context = MagicMock()
        handler.context.config.save_charts = True
        handler.context.logger = MagicMock()  # Mock logger
        handler.context.last_prompt_id = 123
        handler.context.config.save_charts_path = "/custom/path"

        code = 'some text "hello.png" more text'

        code = handler._replace_output_filenames_with_temp_chart(code)

        expected_pattern = re.compile(
            r'some text "exports[/\\]+charts[/\\]+temp_chart_.*\.png" more text'
        )
        self.assertRegex(code, expected_pattern)

    def test_replace_output_filenames_with_temp_chart_windows_paths(self):
        handler = self.cleaner
        handler.context = MagicMock()
        handler.context.config.save_charts = True
        handler.context.logger = MagicMock()
        handler.context.last_prompt_id = 123

        # Use a path with characters that could be escape sequences
        test_dir = os.path.join("C:", "temp", "test", "nested")

        # Create a code string with a filename
        code = 'plt.savefig("original.png")'

        # Replace with our function
        result = handler._replace_output_filenames_with_temp_chart(code)

        # Check that the path is properly formed and doesn't have corruption
        # from escape sequences by extracting the path and trying to use it
        import re

        path_match = re.search(r'"([^"]+)"', result)
        extracted_path = path_match.group(1) if path_match else None

        # Verify the path exists as a string (doesn't have corrupted characters)
        self.assertIsNotNone(extracted_path)

        # On Windows, check that the backslashes are preserved and not interpreted as escapes
        if os.name == "nt":
            # Count backslashes - should be the same as in the directory structure
            # This will fail if "\t" becomes a tab character, etc.
            expected_slashes = (
                test_dir.count("\\") + 2
            )  # +2 for additional path components
            actual_slashes = extracted_path.count("\\")
            self.assertEqual(
                expected_slashes,
                actual_slashes,
                f"Expected {expected_slashes} backslashes but found {actual_slashes}",
            )

    def test_replace_output_filenames_with_temp_chart_empty_code(self):
        handler = self.cleaner

        code = ""
        expected_code = ""  # It should remain empty, as no substitution is made

        result = handler._replace_output_filenames_with_temp_chart(code)

        self.assertEqual(
            result, expected_code, f"Expected '{expected_code}', but got '{result}'"
        )

    def test_replace_output_filenames_with_temp_chart_no_png(self):
        handler = self.cleaner

        code = "some text without png"
        expected_code = "some text without png"  # No change should occur

        result = handler._replace_output_filenames_with_temp_chart(code)

        self.assertEqual(
            result, expected_code, f"Expected '{expected_code}', but got '{result}'"
        )


if __name__ == "__main__":
    unittest.main()
