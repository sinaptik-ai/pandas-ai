import warnings
from typing import Optional

import pandas as pd

from pandasai.data_loader.semantic_layer_schema import SQLConnectionConfig


def load_from_mysql(
    connection_info: SQLConnectionConfig, query: str, params: Optional[list] = None
):
    import pymysql

    conn = pymysql.connect(
        host=connection_info.host,
        user=connection_info.user,
        password=connection_info.password,
        database=connection_info.database,
        port=connection_info.port,
    )
    # Suppress warnings of SqlAlchemy
    # TODO - Later can be removed when SqlAlchemy is to used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.read_sql(query, conn, params=params)


def load_from_postgres(
    connection_info: SQLConnectionConfig, query: str, params: Optional[list] = None
):
    import psycopg2

    conn = psycopg2.connect(
        host=connection_info.host,
        user=connection_info.user,
        password=connection_info.password,
        dbname=connection_info.database,
        port=connection_info.port,
    )
    # Suppress warnings of SqlAlchemy
    # TODO - Later can be removed when SqlAlchemy is to used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.read_sql(query, conn, params=params)


def load_from_cockroachdb(
    connection_info: SQLConnectionConfig, query: str, params: Optional[list] = None
):
    import psycopg2

    conn = psycopg2.connect(
        host=connection_info.host,
        user=connection_info.user,
        password=connection_info.password,
        dbname=connection_info.database,
        port=connection_info.port,
    )
    # Suppress warnings of SqlAlchemy
    # TODO - Later can be removed when SqlAlchemy is to used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.read_sql(query, conn, params=params)


def load_from_sqlserver(
    connection_info: SQLConnectionConfig, query: str, params: Optional[list] = None
):
    import pymssql

    conn = pymssql.connect(
        host=connection_info.host,
        user=connection_info.user,
        password=connection_info.password,
        database=connection_info.database,
        port=connection_info.port,
    )
    # Suppress warnings of SqlAlchemy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.read_sql(query, conn, params=params)


__all__ = [
    "load_from_mysql",
    "load_from_postgres",
    "load_from_cockroachdb",
    "load_from_sqlserver",
]
