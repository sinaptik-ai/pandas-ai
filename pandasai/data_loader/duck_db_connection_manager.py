import logging
import os
import tempfile
import threading
import time
import weakref
from queue import Empty, Queue
from typing import Optional

logger = logging.getLogger(__name__)
import duckdb
from duckdb import DuckDBPyConnection

from pandasai.query_builders.sql_parser import SQLParser


class DuckDBConnectionManager:
    _instance = None
    _lock = threading.Lock()
    _default_pool_size = 60  # Default connection pool size
    _default_max_wait_time = 60  # Default maximum wait time for getting a connection

    def __new__(cls, pool_size=None, max_wait_time=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DuckDBConnectionManager, cls).__new__(cls)
                    # Set pool size and wait time with provided values or defaults
                    cls._instance._pool_size = (
                        pool_size if pool_size is not None else cls._default_pool_size
                    )
                    cls._instance._max_wait_time = (
                        max_wait_time
                        if max_wait_time is not None
                        else cls._default_max_wait_time
                    )
                    cls._instance._init_connection_pool()
                    weakref.finalize(cls._instance, cls._close_connections)
        return cls._instance

    def _init_connection_pool(self):
        """Initialize a DuckDB connection pool."""
        self._connection_pool: Queue[DuckDBPyConnection] = Queue(
            maxsize=self._pool_size
        )
        self._registered_tables = set()
        self._tables_lock = threading.Lock()
        # Create a temporary file with cross-platform compatibility
        temp_dir = tempfile.gettempdir()
        # Add process PID to filename to avoid multi-process conflicts
        self._db_file = os.path.join(temp_dir, f"pandasai_duckdb_temp_{os.getpid()}.db")
        # Create the first connection to initialize the database
        initial_conn = duckdb.connect(self._db_file)
        self._connection_pool.put(initial_conn)
        # Create remaining connections to the same database file
        for _ in range(self._pool_size - 1):
            conn = duckdb.connect(self._db_file)
            self._connection_pool.put(conn)

    def _get_connection(self):
        """Get a connection from the pool."""
        try:
            return self._connection_pool.get(timeout=self._max_wait_time)
        except Empty:
            raise RuntimeError("No available connections in the pool")

    def _release_connection(self, conn):
        """Release a connection back to the pool."""
        self._connection_pool.put(conn)

    @classmethod
    def _close_connections(cls):
        """Closes all connections in the pool when the instance is deleted."""
        if cls._instance:
            with cls._lock:
                # Close all connections
                while not cls._instance._connection_pool.empty():
                    conn = cls._instance._connection_pool.get()
                    conn.close()
                # Remove the temporary database file
                if hasattr(cls._instance, "_db_file") and os.path.exists(
                    cls._instance._db_file
                ):
                    try:
                        os.remove(cls._instance._db_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary database file: {e}")
                cls._instance = None

    def register(self, name: str, df):
        """Registers a DataFrame as a DuckDB table.

        Args:
            name: Table name to register
            df: DataFrame to register as table

        Note:
            This method is thread-safe and handles concurrent table creation attempts.
            If multiple threads try to create the same table simultaneously,
            only one will succeed and others will reuse the existing table.
        """
        # First check if table is already registered in our tracking set
        with self._tables_lock:
            if name in self._registered_tables:
                return  # Table already exists, no need to register again

        conn = self._get_connection()
        try:
            # Register the DataFrame as a temporary view
            conn.register(name, df)

            # Try to create the table with retry logic for concurrent scenarios
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Start a new transaction for each attempt
                    conn.begin()
                    try:
                        # Use CREATE TABLE IF NOT EXISTS to handle race conditions
                        conn.execute(
                            f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM {name}"
                        )
                        conn.commit()
                        with self._tables_lock:
                            self._registered_tables.add(name)
                        break  # Success, exit retry loop
                    except Exception as e:
                        # Explicit rollback on any error
                        conn.rollback()
                        raise
                except duckdb.TransactionException as e:
                    if (
                        "Catalog write-write conflict" in str(e)
                        and attempt < max_retries - 1
                    ):
                        # Wait a bit before retrying (exponential backoff)
                        time.sleep(0.1 * (2**attempt))
                        continue
                    elif "already exists" in str(e):
                        # Table was created by another thread, consider this success
                        with self._tables_lock:
                            self._registered_tables.add(name)
                        break
                    raise  # Re-raise other transaction exceptions
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise  # Re-raise on last attempt
                    time.sleep(0.1 * (attempt + 1))
        finally:
            self._release_connection(conn)

    def sql(self, query: str, params: Optional[list] = None):
        """Executes an SQL query and returns the result as a Pandas DataFrame."""
        conn = self._get_connection()
        try:
            query = SQLParser.transpile_sql_dialect(query, to_dialect="duckdb")
            result = conn.sql(query, params=params)
        finally:
            self._release_connection(conn)
        return result

    def close(self):
        """Manually close the connection if needed."""
        self._close_connections()
