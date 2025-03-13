from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from pandasai.data_loader.loader import DatasetLoader
from pandasai.data_loader.local_loader import LocalDatasetLoader
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import MaliciousQueryError
from pandasai.query_builders import LocalQueryBuilder


class TestDatasetLoader:
    def test_load_from_local_source_valid(self, sample_schema):
        with patch(
            "pandasai.data_loader.local_loader.LocalDatasetLoader.execute_query"
        ) as mock_execute_query_builder:
            sample_schema.transformations = None
            loader = LocalDatasetLoader(sample_schema, "test/test")

            mock_execute_query_builder.return_value = DataFrame(
                {"email": ["test@example.com"]}
            )

            result = loader.load()

            assert isinstance(result, DataFrame)
            mock_execute_query_builder.assert_called_once()
            assert "email" in result.columns

    def test_local_loader_properties(self, sample_schema):
        loader = LocalDatasetLoader(sample_schema, "test/test")
        assert isinstance(loader.query_builder, LocalQueryBuilder)

    def test_load_schema_mysql_invalid_name(self, mysql_schema):
        mysql_schema.name = "invalid-name"

        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=str(mysql_schema.to_yaml()))
        ):
            with pytest.raises(
                ValueError,
                match="Dataset name must be lowercase and use underscores instead of spaces.",
            ):
                DatasetLoader._read_schema_file("test/users")

    def test_load_from_local_source_invalid_source_type(self, sample_schema):
        sample_schema.source.type = "mysql"
        loader = LocalDatasetLoader(sample_schema, "test/test")

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load()

    def test_load_schema(self, sample_schema):
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=str(sample_schema.to_yaml()))
        ):
            schema = DatasetLoader._read_schema_file("test/users")
            assert schema == sample_schema

    def test_load_schema_mysql(self, mysql_schema):
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=str(mysql_schema.to_yaml()))
        ):
            schema = DatasetLoader._read_schema_file("test/users")
            assert schema == mysql_schema

    def test_load_schema_file_not_found(self):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                DatasetLoader._read_schema_file("test/users")

    def test_read_file(self, sample_schema):
        sample_schema.transformations = None
        loader = LocalDatasetLoader(sample_schema, "test/test")

        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        with patch(
            "pandasai.data_loader.local_loader.LocalDatasetLoader.execute_query"
        ) as mock_execute_query_builder:
            mock_execute_query_builder.return_value = mock_df
            result = loader.load()
            mock_execute_query_builder.assert_called_once()
            assert isinstance(result, pd.DataFrame)
            assert result.equals(mock_df)

    def test_build_dataset_csv_schema(self, sample_schema):
        """Test loading data from a CSV schema directly and creates a VirtualDataFrame and handles queries correctly."""
        with patch("os.path.exists", return_value=True), patch(
            "pandasai.data_loader.local_loader.LocalDatasetLoader.execute_query"
        ) as mock_execute_query:
            sample_schema.transformations = None
            mock_data = {
                "email": ["test@example.com"],
                "first_name": ["John"],
                "timestamp": ["2023-01-01"],
            }
            mock_execute_query.return_value = DataFrame(mock_data)
            loader = LocalDatasetLoader(sample_schema, "test/test")

            result = loader.load()

            assert isinstance(result, DataFrame)
            assert "email" in result.columns

    def test_malicious_query(self, sample_schema):
        loader = LocalDatasetLoader(sample_schema, "test/test")
        with pytest.raises(MaliciousQueryError):
            loader.execute_query("DROP TABLE")

    def test_runtime_error(self, sample_schema):
        loader = LocalDatasetLoader(sample_schema, "test/test")
        with pytest.raises(RuntimeError):
            loader.execute_query("SELECT * FROM nonexistent_table")

    def test_read_parquet_file(self, sample_schema):
        loader = LocalDatasetLoader(sample_schema, "test/test")
        with pytest.raises(RuntimeError):
            loader.execute_query(
                """SELECT
            "*",
            FROM READ_PARQUET(
            'http://127.0.0.1:54321/storage/v1/object/sign/datasets/pai-personal-32771/spf-base/data.parquet?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJkYXRhc2V0cy9wYWktcGVyc29uYWwtMzI3NzEvaGEzMDIwZS1jbGktc3BmLWJhc2UvZGF0YS5wYXJxdWV0IiwiaWF0IjoxNzQxODcwMTI3LCJleHAiOjE3NDE4NzAxNTd9.pzCL4efZJbZiAXzzbjFEiI--a3WAwECYzKhMwF3r5vE'
            )"""
            )

    def test_read_parquet_file_with_mock_query_validator(self, sample_schema):
        with patch("os.path.exists", return_value=True), patch(
            "pandasai.data_loader.local_loader.is_sql_query_safe"
        ) as mock_is_query_safe:
            loader = LocalDatasetLoader(sample_schema, "test/test")
            with pytest.raises(RuntimeError):
                loader.execute_query(
                    """SELECT
                "*",
                FROM READ_PARQUET(
                'http://127.0.0.1:54321/storage/v1/object/sign/datasets/pai-personal-32771/spf-base/data.parquet?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJkYXRhc2V0cy9wYWktcGVyc29uYWwtMzI3NzEvaGEzMDIwZS1jbGktc3BmLWJhc2UvZGF0YS5wYXJxdWV0IiwiaWF0IjoxNzQxODcwMTI3LCJleHAiOjE3NDE4NzAxNTd9.pzCL4efZJbZiAXzzbjFEiI--a3WAwECYzKhMwF3r5vE'
                )"""
                )

                mock_is_query_safe.assert_called_once_with(
                    """SELECT
                "*",
                FROM dummy_table"""
                )
