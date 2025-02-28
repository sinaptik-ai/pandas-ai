import weakref
from typing import Optional

import duckdb

from pandasai.query_builders.sql_parser import SQLParser


class DuckDBConnectionManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DuckDBConnectionManager, cls).__new__(cls)
            cls._instance._init_connection()
            weakref.finalize(cls._instance, cls._close_connection)
        return cls._instance

    def _init_connection(self):
        """Initialize a DuckDB connection."""
        self.connection = duckdb.connect()
        self._registered_tables = set()

    @classmethod
    def _close_connection(cls):
        """Closes the DuckDB connection when the instance is deleted."""
        if cls._instance and hasattr(cls._instance, "connection"):
            cls._instance.connection.close()
            cls._instance = None

    def register(self, name: str, df):
        """Registers a DataFrame as a DuckDB table."""
        self.connection.register(name, df)
        self._registered_tables.add(name)

    def sql(self, query: str, params: Optional[list] = None):
        """Executes an SQL query and returns the result as a Pandas DataFrame."""
        query = SQLParser.transpile_sql_dialect(query, to_dialect="duckdb")
        return self.connection.sql(query, params=params)

    def close(self):
        """Manually close the connection if needed."""
        self._close_connection()
