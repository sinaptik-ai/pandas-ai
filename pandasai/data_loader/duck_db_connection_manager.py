from typing import Optional

import duckdb

from pandasai.query_builders.sql_parser import SQLParser


class DuckDBConnectionManager:
    def __init__(self):
        """Initialize a DuckDB connection."""
        self.connection = duckdb.connect()
        self._registered_tables = set()

    def __del__(self):
        """Destructor to ensure the DuckDB connection is closed."""
        self.close()

    def register(self, name: str, df):
        """Registers a DataFrame as a DuckDB table."""
        self.connection.register(name, df)
        self._registered_tables.add(name)

    def unregister(self, name: str):
        """Unregister a previously registered DuckDB table."""
        if name in self._registered_tables:
            self.connection.unregister(name)
            self._registered_tables.remove(name)

    def sql(self, query: str, params: Optional[list] = None):
        """Executes an SQL query and returns the result as a Pandas DataFrame."""
        query = SQLParser.transpile_sql_dialect(query, to_dialect="duckdb")
        return self.connection.sql(query, params=params)

    def close(self):
        """Closes the DuckDB connection."""
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
            self.connection = None
            self._registered_tables.clear()
