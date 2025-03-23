import re
from typing import Optional

import duckdb
import pandas as pd

from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import MaliciousQueryError
from pandasai.query_builders import LocalQueryBuilder

from ..helpers.sql_sanitizer import is_sql_query_safe
from .duck_db_connection_manager import DuckDBConnectionManager
from .loader import DatasetLoader
from .semantic_layer_schema import SemanticLayerSchema


class LocalDatasetLoader(DatasetLoader):
    """
    Loader for local datasets (CSV, Parquet).
    """

    def __init__(self, schema: SemanticLayerSchema, dataset_path: str):
        super().__init__(schema, dataset_path)
        self._query_builder: LocalQueryBuilder = LocalQueryBuilder(schema, dataset_path)

    @property
    def query_builder(self) -> LocalQueryBuilder:
        return self._query_builder

    def register_table(self):
        df = self.load()
        db_manager = DuckDBConnectionManager()
        db_manager.register(self.schema.name, df)

    def load(self) -> DataFrame:
        df: pd.DataFrame = self.execute_query(self.query_builder.build_query())
        return DataFrame(
            df,
            schema=self.schema,
            path=self.dataset_path,
        )

    def _replace_readparquet_block_with_table(
        self, sql_query, table: str = "dummy_table"
    ):
        read_parquet_pattern = re.compile(r"(READ_PARQUET\(\s*'[^']+'\s*\))", re.DOTALL)
        read_parquet_blocks = read_parquet_pattern.findall(sql_query)
        for block in read_parquet_blocks:
            sql_query = sql_query.replace(block, table)

        return sql_query

    def execute_query(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
        try:
            db_manager = DuckDBConnectionManager()

            # Replace READ_PARQUET blocks with a dummy table for validation
            validation_query = self._replace_readparquet_block_with_table(query)

            if not is_sql_query_safe(validation_query, dialect="duckdb"):
                raise MaliciousQueryError(
                    "The SQL query is deemed unsafe and will not be executed."
                )

            return db_manager.sql(query, params=params).df()
        except duckdb.Error as e:
            raise RuntimeError(f"SQL execution failed: {e}") from e
