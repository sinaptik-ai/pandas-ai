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

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            db_manager = DuckDBConnectionManager()

            if not is_sql_query_safe(query, dialect="duckdb"):
                raise MaliciousQueryError(
                    "The SQL query is deemed unsafe and will not be executed."
                )

            return db_manager.sql(query).df()
        except duckdb.Error as e:
            raise RuntimeError(f"SQL execution failed: {e}") from e
