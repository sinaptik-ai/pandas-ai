from unittest.mock import MagicMock, mock_open, patch

import pytest
import sqlglot

from pandasai.data_loader.semantic_layer_schema import (
    SemanticLayerSchema,
    Transformation,
)
from pandasai.query_builders import LocalQueryBuilder
from pandasai.query_builders.base_query_builder import BaseQueryBuilder
from pandasai.query_builders.sql_query_builder import SqlQueryBuilder


class TestQueryBuilder:
    @pytest.fixture
    def mysql_schema(self):
        raw_schema = {
            "name": "users",
            "update_frequency": "weekly",
            "columns": [
                {
                    "name": "email",
                    "type": "string",
                    "description": "User's email address",
                },
                {
                    "name": "first_name",
                    "type": "string",
                    "description": "User's first name",
                },
                {
                    "name": "timestamp",
                    "type": "datetime",
                    "description": "Timestamp of the record",
                },
            ],
            "order_by": ["created_at DESC"],
            "limit": 100,
            "source": {
                "type": "mysql",
                "connection": {
                    "host": "localhost",
                    "port": 3306,
                    "database": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                },
                "table": "users",
            },
        }
        return SemanticLayerSchema(**raw_schema)

    def test_build_query_csv(self, sample_schema):
        with patch(
            "pandasai.query_builders.local_query_builder.ConfigManager.get"
        ) as mock_config_get:
            # Mock the return of `ConfigManager.get()`
            mock_config = MagicMock()
            mock_config.file_manager.abs_path.return_value = "/mocked/absolute/path"
            mock_config_get.return_value = mock_config
            query_builder = LocalQueryBuilder(sample_schema, "test/test")
            query = query_builder.build_query()
            expected_query = (
                "SELECT\n"
                '  "email",\n'
                '  "first_name",\n'
                '  "timestamp"\n'
                "FROM READ_CSV('/mocked/absolute/path')\n"
                "ORDER BY\n"
                '  "created_at" DESC\n'
                "LIMIT 100"
            )
            assert query == expected_query

    def test_build_query_csv_with_transformation(self, raw_sample_schema):
        with patch(
            "pandasai.query_builders.local_query_builder.ConfigManager.get"
        ) as mock_config_get:
            # Mock the return of `ConfigManager.get()`
            raw_sample_schema["transformations"] = [
                {"type": "anonymize", "params": {"column": "email"}},
                {
                    "type": "convert_timezone",
                    "params": {"column": "timestamp", "to": "UTC"},
                },
            ]
            sample_schema = SemanticLayerSchema(**raw_sample_schema)
            mock_config = MagicMock()
            mock_config.file_manager.abs_path.return_value = "/mocked/absolute/path"
            mock_config_get.return_value = mock_config
            query_builder = LocalQueryBuilder(sample_schema, "test/test")
            query = query_builder.build_query()
            expected_query = (
                "SELECT\n"
                '  MD5("email") AS "email",\n'
                '  "first_name" AS "first_name",\n'
                "  CONVERT_TZ(\"timestamp\", 'UTC', 'UTC') AS \"timestamp\"\n"
                "FROM READ_CSV('/mocked/absolute/path')\n"
                "ORDER BY\n"
                '  "created_at" DESC\n'
                "LIMIT 100"
            )
            assert query == expected_query

    def test_build_query_parquet(self, sample_schema):
        sample_schema.source.type = "parquet"
        with patch(
            "pandasai.query_builders.local_query_builder.ConfigManager.get"
        ) as mock_config_get:
            # Mock the return of `ConfigManager.get()`
            mock_config = MagicMock()
            mock_config.file_manager.abs_path.return_value = "/mocked/absolute/path"
            mock_config_get.return_value = mock_config
            query_builder = LocalQueryBuilder(sample_schema, "test/test")
            query = query_builder.build_query()
            expected_query = (
                "SELECT\n"
                '  "email",\n'
                '  "first_name",\n'
                '  "timestamp"\n'
                "FROM READ_PARQUET('/mocked/absolute/path')\n"
                "ORDER BY\n"
                '  "created_at" DESC\n'
                "LIMIT 100"
            )
            assert query == expected_query

    def test_build_query(self, mysql_schema):
        query_builder = SqlQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        expected_query = (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )
        assert query == expected_query

    def test_build_query_with_transformation(self, raw_mysql_schema):
        raw_mysql_schema["transformations"] = [
            {"type": "anonymize", "params": {"column": "email"}},
            {
                "type": "convert_timezone",
                "params": {"column": "timestamp", "to": "UTC"},
            },
        ]
        mysql_schema = SemanticLayerSchema(**raw_mysql_schema)
        query_builder = SqlQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        expected_query = (
            "SELECT\n"
            '  MD5("email") AS "email",\n'
            '  "first_name" AS "first_name",\n'
            "  CONVERT_TZ(\"timestamp\", 'UTC', 'UTC') AS \"timestamp\"\n"
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )
        assert query == expected_query

    def test_build_query_invalid(self, mysql_schema):
        mysql_schema.columns = ["invalid"]
        query_builder = SqlQueryBuilder(mysql_schema)
        with pytest.raises(
            ValueError,
            match="Failed to generate a valid SQL query from the provided schema:",
        ):
            query_builder.validate_query_builder()

    def test_build_query_without_order_by(self, mysql_schema):
        mysql_schema.order_by = None
        query_builder = SqlQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        expected_query = 'SELECT\n  "email",\n  "first_name",\n  "timestamp"\nFROM "users"\nLIMIT 100'
        assert query == expected_query

    def test_build_query_without_limit(self, mysql_schema):
        mysql_schema.limit = None
        query_builder = SqlQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        expected_query = (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC'
        )
        assert query == expected_query

    def test_build_query_with_multiple_order_by(self, mysql_schema):
        mysql_schema.order_by = ["created_at DESC", "email ASC"]
        query_builder = SqlQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        expected_query = (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC,\n'
            '  "email" ASC\n'
            "LIMIT 100"
        )
        assert query == expected_query

    def test_table_name_injection(self, mysql_schema):
        mysql_schema.name = "users; DROP TABLE users;"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users; DROP TABLE users;"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_column_name_injection(self, mysql_schema):
        mysql_schema.columns[0].name = "column; DROP TABLE users;"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "column; DROP TABLE users;",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_table_name_union_injection(self, mysql_schema):
        mysql_schema.name = "users UNION SELECT 1,2,3;"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users UNION SELECT 1,2,3;"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_column_name_union_injection(self, mysql_schema):
        mysql_schema.columns[
            0
        ].name = "column UNION SELECT username, password FROM users;"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "column UNION SELECT username, password FROM users;",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_table_name_comment_injection(self, mysql_schema):
        mysql_schema.name = "users --"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_column_name_comment_injection(self, mysql_schema):
        mysql_schema.columns[0].name = "column --"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "column",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_table_name_stacked_query_injection(self, mysql_schema):
        mysql_schema.name = 'users"; SELECT * FROM sensitive_data; --'
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users""; SELECT * FROM sensitive_data; --"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_table_name_batch_injection(self, mysql_schema):
        mysql_schema.name = "users; TRUNCATE users; SELECT * FROM users WHERE 't'='t"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            "FROM \"users; TRUNCATE users; SELECT * FROM users WHERE 't'='t\"\n"
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    def test_table_name_time_based_injection(self, mysql_schema):
        mysql_schema.name = "users' AND (SELECT * FROM (SELECT(SLEEP(5)))test); --"
        query_builder = BaseQueryBuilder(mysql_schema)
        query = query_builder.build_query()
        assert query == (
            "SELECT\n"
            '  "email",\n'
            '  "first_name",\n'
            '  "timestamp"\n'
            'FROM "users\' AND (SELECT * FROM (SELECT(SLEEP(5)))test); --"\n'
            "ORDER BY\n"
            '  "created_at" DESC\n'
            "LIMIT 100"
        )

    @pytest.mark.parametrize(
        "injection",
        [
            "users; DROP TABLE users;",
            "users UNION SELECT 1,2,3;",
            'users"; SELECT * FROM sensitive_data; --',
            "users; TRUNCATE users; SELECT * FROM users WHERE 't'='t",
            "users' AND (SELECT * FROM (SELECT(SLEEP(5)))test); --",
        ],
    )
    def test_order_by_injection(self, injection, mysql_schema):
        mysql_schema.order_by = [injection]
        query_builder = BaseQueryBuilder(mysql_schema)
        with pytest.raises((sqlglot.errors.ParseError, sqlglot.errors.TokenError)):
            query_builder.build_query()

    def test_build_query_distinct(self, sample_schema):
        base_query_builder = BaseQueryBuilder(sample_schema)
        base_query_builder.schema.transformations = [
            Transformation(type="remove_duplicates")
        ]
        result = base_query_builder.build_query()
        assert result.startswith("SELECT DISTINCT")

    def test_build_query_distinct_head(self, sample_schema):
        base_query_builder = BaseQueryBuilder(sample_schema)
        base_query_builder.schema.transformations = [
            Transformation(type="remove_duplicates")
        ]
        result = base_query_builder.get_head_query()
        assert result.startswith("SELECT DISTINCT")

    def test_build_query_order_by(self, sample_schema):
        base_query_builder = BaseQueryBuilder(sample_schema)
        base_query_builder.schema.order_by = ["column"]
        result = base_query_builder.build_query()
        assert 'ORDER BY\n  "column"' in result

    def test_get_group_by_columns(self, sample_schema):
        base_query_builder = BaseQueryBuilder(sample_schema)
        base_query_builder.schema.group_by = ["parents"]
        result = base_query_builder.get_head_query()
        assert 'GROUP BY\n  "parents"' in result
