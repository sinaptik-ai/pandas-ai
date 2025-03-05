import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pandasai import VirtualDataFrame
from pandasai.data_loader.sql_loader import SQLDatasetLoader
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import MaliciousQueryError


class TestSqlDatasetLoader:
    def test_load_mysql_source(self, mysql_schema):
        """Test loading data from a MySQL source creates a VirtualDataFrame and handles queries correctly."""
        with patch(
            "pandasai.data_loader.sql_loader.SQLDatasetLoader.execute_query"
        ) as mock_execute_query:
            # Mock the query results
            mock_execute_query.return_value = DataFrame(
                pd.DataFrame(
                    {
                        "email": ["test@example.com"],
                        "first_name": ["John"],
                        "timestamp": [pd.Timestamp.now()],
                    }
                )
            )

            loader = SQLDatasetLoader(mysql_schema, "test/users")
            result = loader.load()

            # Test that we get a VirtualDataFrame
            assert isinstance(result, DataFrame)
            assert result.schema == mysql_schema

            # Test that load_head() works
            head_result = result.head()
            assert isinstance(head_result, DataFrame)
            assert "email" in head_result.columns
            assert "first_name" in head_result.columns
            assert "timestamp" in head_result.columns

            # Verify the SQL query was executed correctly
            mock_execute_query.assert_called_once_with(
                'SELECT\n  "email",\n  "first_name",\n  "timestamp"\nFROM "users"\nLIMIT 5'
            )

            # Test executing a custom query
            custom_query = "SELECT email FROM users WHERE first_name = 'John'"
            result.execute_sql_query(custom_query)
            mock_execute_query.assert_called_with(custom_query)

    def test_mysql_malicious_query(self, mysql_schema):
        """Test loading data from a MySQL source creates a VirtualDataFrame and handles queries correctly."""
        with patch(
            "pandasai.data_loader.sql_loader.is_sql_query_safe"
        ) as mock_sql_query, patch(
            "pandasai.data_loader.sql_loader.SQLDatasetLoader._get_loader_function"
        ) as mock_loader_function:
            mocked_exec_function = MagicMock()
            mock_df = DataFrame(
                pd.DataFrame(
                    {
                        "email": ["test@example.com"],
                        "first_name": ["John"],
                        "timestamp": [pd.Timestamp.now()],
                    }
                )
            )
            mocked_exec_function.return_value = mock_df
            mock_loader_function.return_value = mocked_exec_function
            loader = SQLDatasetLoader(mysql_schema, "test/users")
            mock_sql_query.return_value = False
            logging.debug("Loading schema from dataset path: %s", loader)

            with pytest.raises(MaliciousQueryError):
                loader.execute_query("DROP TABLE users")

            mock_sql_query.assert_called_once_with("DROP TABLE users", "mysql")

    def test_mysql_safe_query(self, mysql_schema):
        """Test loading data from a MySQL source creates a VirtualDataFrame and handles queries correctly."""
        with patch(
            "pandasai.data_loader.sql_loader.is_sql_query_safe"
        ) as mock_sql_query, patch(
            "pandasai.data_loader.sql_loader.SQLDatasetLoader._get_loader_function"
        ) as mock_loader_function:
            mocked_exec_function = MagicMock()
            mock_df = DataFrame(
                pd.DataFrame(
                    {
                        "email": ["test@example.com"],
                        "first_name": ["John"],
                        "timestamp": [pd.Timestamp.now()],
                    }
                )
            )
            mocked_exec_function.return_value = mock_df
            mock_loader_function.return_value = mocked_exec_function
            loader = SQLDatasetLoader(mysql_schema, "test/users")
            mock_sql_query.return_value = True
            logging.debug("Loading schema from dataset path: %s", loader)

            result = loader.execute_query("SELECT * FROM users")

            assert isinstance(result, DataFrame)
            mock_sql_query.assert_called_once_with("SELECT\n  *\nFROM users", "mysql")

    def test_mysql_malicious_with_no_import(self, mysql_schema):
        """Test loading data from a MySQL source creates a VirtualDataFrame and handles queries correctly."""
        with patch(
            "pandasai.data_loader.sql_loader.is_sql_query_safe"
        ) as mock_sql_query, patch(
            "pandasai.data_loader.sql_loader.SQLDatasetLoader._get_loader_function"
        ) as mock_loader_function:
            mocked_exec_function = MagicMock()
            mock_df = DataFrame(
                pd.DataFrame(
                    {
                        "email": ["test@example.com"],
                        "first_name": ["John"],
                        "timestamp": [pd.Timestamp.now()],
                    }
                )
            )
            mocked_exec_function.return_value = mock_df

            mock_exec_function = MagicMock()
            mock_loader_function.return_value = mock_exec_function
            mock_exec_function.side_effect = ModuleNotFoundError("Error")
            loader = SQLDatasetLoader(mysql_schema, "test/users")
            mock_sql_query.return_value = True
            logging.debug("Loading schema from dataset path: %s", loader)
            with pytest.raises(ImportError):
                loader.execute_query("select * from users")
