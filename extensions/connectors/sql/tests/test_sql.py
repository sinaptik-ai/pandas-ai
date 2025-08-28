import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Assuming the functions are in a module called db_loader
from pandasai_sql import (
    load_from_cockroachdb,
    load_from_mysql,
    load_from_postgres,
    load_from_sqlserver,
)

from pandasai.data_loader.semantic_layer_schema import SQLConnectionConfig


class TestDatabaseLoader(unittest.TestCase):
    @patch("pymysql.connect")
    @patch("pandas.read_sql")
    def test_load_from_mysql(self, mock_read_sql, mock_pymysql_connect):
        # Setup the mock return values
        mock_conn = MagicMock()
        mock_pymysql_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [1, 2], "column2": [3, 4]}
        )

        # Test data
        connection_info = {
            "host": "localhost",
            "user": "root",
            "password": "password",
            "database": "test_db",
            "port": 3306,
        }

        query = "SELECT * FROM test_table"

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_mysql(connection_config, query)

        # Assert that the connection is made and SQL query is executed
        mock_pymysql_connect.assert_called_once_with(
            host="localhost",
            user="root",
            password="password",
            database="test_db",
            port=3306,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=None)

        # Assert the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("psycopg2.connect")
    @patch("pandas.read_sql")
    def test_load_from_postgres(self, mock_read_sql, mock_psycopg2_connect):
        # Setup the mock return values
        mock_conn = MagicMock()
        mock_psycopg2_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [5, 6], "column2": [7, 8]}
        )

        # Test data
        connection_info = {
            "host": "localhost",
            "user": "postgres",
            "password": "password",
            "database": "test_db",
            "port": 5432,
        }
        query = "SELECT * FROM test_table"

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_postgres(connection_config, query)

        # Assert that the connection is made and SQL query is executed
        mock_psycopg2_connect.assert_called_once_with(
            host="localhost",
            user="postgres",
            password="password",
            dbname="test_db",
            port=5432,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=None)

        # Assert the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("psycopg2.connect")
    @patch("pandas.read_sql")
    def test_load_from_cockroachdb(self, mock_read_sql, mock_postgresql_connect):
        # Setup the mock return values
        mock_conn = MagicMock()
        mock_postgresql_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [13, 14], "column2": [15, 16]}
        )

        # Test data
        connection_info = {
            "host": "localhost",
            "user": "root",
            "password": "password",
            "database": "test_db",
            "port": 26257,
        }
        query = "SELECT * FROM test_table"

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_cockroachdb(connection_config, query)

        # Assert that the connection is made and SQL query is executed
        mock_postgresql_connect.assert_called_once_with(
            host="localhost",
            user="root",
            password="password",
            dbname="test_db",
            port=26257,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=None)

        # Assert the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("pymysql.connect")
    @patch("pandas.read_sql")
    def test_load_from_mysql_with_params(self, mock_read_sql, mock_pymysql_connect):
        mock_conn = MagicMock()
        mock_pymysql_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [1, 2], "column2": [3, 4]}
        )

        connection_info = {
            "host": "localhost",
            "user": "root",
            "password": "password",
            "database": "test_db",
            "port": 3306,
        }
        query = "SELECT * FROM test_table WHERE id = %s"
        query_params = [123]

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_mysql(connection_config, query, query_params)

        mock_pymysql_connect.assert_called_once_with(
            host="localhost",
            user="root",
            password="password",
            database="test_db",
            port=3306,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=query_params)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("psycopg2.connect")
    @patch("pandas.read_sql")
    def test_load_from_postgres_with_params(self, mock_read_sql, mock_psycopg2_connect):
        mock_conn = MagicMock()
        mock_psycopg2_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [5, 6], "column2": [7, 8]}
        )

        connection_info = {
            "host": "localhost",
            "user": "postgres",
            "password": "password",
            "database": "test_db",
            "port": 5432,
        }
        query = "SELECT * FROM test_table WHERE name ILIKE %s"
        query_params = ["%John%"]

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_postgres(connection_config, query, query_params)

        mock_psycopg2_connect.assert_called_once_with(
            host="localhost",
            user="postgres",
            password="password",
            dbname="test_db",
            port=5432,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=query_params)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("psycopg2.connect")
    @patch("pandas.read_sql")
    def test_load_from_cockroachdb_with_params(
        self, mock_read_sql, mock_postgresql_connect
    ):
        mock_conn = MagicMock()
        mock_postgresql_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [13, 14], "column2": [15, 16]}
        )

        connection_info = {
            "host": "localhost",
            "user": "root",
            "password": "password",
            "database": "test_db",
            "port": 26257,
        }
        query = "SELECT * FROM test_table WHERE status = %s"
        query_params = ["active"]

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_cockroachdb(connection_config, query, query_params)

        mock_postgresql_connect.assert_called_once_with(
            host="localhost",
            user="root",
            password="password",
            dbname="test_db",
            port=26257,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=query_params)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("pymssql.connect")
    @patch("pandas.read_sql")
    def test_load_from_sqlserver(self, mock_read_sql, mock_pymssql_connect):
        # Setup the mock return values
        mock_conn = MagicMock()
        mock_pymssql_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [9, 10], "column2": [11, 12]}
        )

        # Test data
        connection_info = {
            "host": "localhost",
            "user": "sa",
            "password": "password",
            "database": "test_db",
            "port": 1433,
        }
        query = "SELECT * FROM test_table"

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_sqlserver(connection_config, query)

        # Assert that the connection is made and SQL query is executed
        mock_pymssql_connect.assert_called_once_with(
            host="localhost",
            user="sa",
            password="password",
            database="test_db",
            port=1433,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=None)

        # Assert the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("pymssql.connect")
    @patch("pandas.read_sql")
    def test_load_from_sqlserver_with_params(self, mock_read_sql, mock_pymssql_connect):
        mock_conn = MagicMock()
        mock_pymssql_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(
            {"column1": [9, 10], "column2": [11, 12]}
        )

        connection_info = {
            "host": "localhost",
            "user": "sa",
            "password": "password",
            "database": "test_db",
            "port": 1433,
        }
        query = "SELECT * FROM test_table WHERE id = %s"
        query_params = [456]

        connection_config = SQLConnectionConfig(**connection_info)

        result = load_from_sqlserver(connection_config, query, query_params)

        mock_pymssql_connect.assert_called_once_with(
            host="localhost",
            user="sa",
            password="password",
            database="test_db",
            port=1433,
        )
        mock_read_sql.assert_called_once_with(query, mock_conn, params=query_params)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
