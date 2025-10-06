# -*- coding: utf-8 -*-
"""
PandasAI is a wrapper around a LLM to make dataframes conversational
"""
from __future__ import annotations

import os
from io import BytesIO
from typing import Hashable, List, Optional, Union

import pandas as pd

from pandasai.config import APIKeyManager, ConfigManager
from pandasai.data_loader.semantic_layer_schema import (
    Column,
    Relation,
    SemanticLayerSchema,
    Source,
    Transformation,
)
from pandasai.ee.skills import skill
from pandasai.ee.skills.manager import SkillsManager
from pandasai.exceptions import DatasetNotFound, InvalidConfigError
from pandasai.helpers.path import (
    find_project_root,
    get_validated_dataset_path,
    transform_dash_to_underscore,
)
from pandasai.sandbox.sandbox import Sandbox

from .agent import Agent
from .data_loader.loader import DatasetLoader
from .data_loader.semantic_layer_schema import (
    Column,
)
from .dataframe import DataFrame, VirtualDataFrame
from .helpers.path import get_table_name_from_path
from .helpers.sql_sanitizer import (
    sanitize_sql_table_name,
    sanitize_sql_table_name_lowercase,
)
from .smart_dataframe import SmartDataframe
from .smart_datalake import SmartDatalake


def create(
    path: str,
    df: Optional[DataFrame] = None,
    description: Optional[str] = None,
    columns: Optional[List[dict]] = None,
    source: Optional[dict] = None,
    relations: Optional[List[dict]] = None,
    view: bool = False,
    group_by: Optional[List[str]] = None,
    transformations: Optional[List[dict]] = None,
) -> Union[DataFrame, VirtualDataFrame]:
    """
    Creates a new dataset at the specified path with optional metadata, schema,
    and data source configurations.

    Args:
        path (str): Path in the format 'organization/dataset'. Specifies the location
            where the dataset should be created. The organization and dataset names
            must be lowercase, with hyphens instead of spaces.
        df (DataFrame, optional): The DataFrame containing the data to save. If not
            provided, a connector must be specified to define the dataset source.
        description (str, optional): A textual description of the dataset. Defaults
            to None.
        columns (List[dict], optional): A list of dictionaries defining the column schema.
            Each dictionary should include keys such as 'name', 'type', and optionally
            'description' to describe individual columns. If not provided, the schema
            will be inferred from the DataFrame or connector.
        source (dict, optional): A dictionary specifying the data source configuration.
            Required if `df` is not provided. The connector may include keys like 'type',
            'table', or 'view' to define the data source type and structure.
        relations (dict, optional): A dictionary specifying relationships between tables
            when the dataset is created as a view. Each relationship should be defined
            using keys such as 'type', 'source', and 'target'.
        view (bool, optional): If True, the dataset will be created as a view instead
        group_by (List[str], optional): A list of column names to use for grouping in SQL
            queries. Each column name should correspond to a non-aggregated column in the
            dataset. Aggregated columns (those with expressions) cannot be included in
            group_by.
        transformations (List[dict], optional): A list of transformation dictionaries

    Returns:
        Union[DataFrame, VirtualDataFrame]: The created dataset object. This may be
        a physical DataFrame if data is saved locally, or a VirtualDataFrame if
        defined using a connector or relations.

    Raises:
        ValueError: If the `path` format is invalid, the organization or dataset
            name contains unsupported characters, or a dataset already exists at
            the specified path.
        InvalidConfigError: If neither `df` nor a valid `source` is provided.

    Examples:
        >>> # Create a simple dataset
        >>> create(
        ...     path="my-org/my-dataset",
        ...     df=my_dataframe,
        ...     description="This is a sample dataset.",
        ...     columns=[
        ...         {"name": "id", "type": "integer", "description": "Primary key"},
        ...         {"name": "name", "type": "string", "description": "Name of the item"},
        ...     ],
        ... )
        Dataset saved successfully to path: datasets/my-org/my-dataset

        >>> # Create a dataset with transformations and group by
        >>> create(
        ...     path="my-org/sales",
        ...     df=sales_df,
        ...     description="Sales data with transformations",
        ...     columns=[
        ...         {"name": "category", "type": "string", "description": "Product category"},
        ...         {"name": "region", "type": "string", "description": "Sales region"},
        ...         {"name": "amount", "type": "float", "expression": "sum(amount)", "alias": "total_sales"},
        ...         {"name": "quantity", "type": "integer", "expression": "avg(quantity)", "alias": "avg_quantity"},
        ...     ],
        ...     transformations=[
        ...         {
        ...             "type": "fill_na",
        ...             "params": {"column": "amount", "value": 0}
        ...         },
        ...         {
        ...             "type": "map_values",
        ...             "params": {
        ...                 "column": "category",
        ...                 "mapping": {"A": "Premium", "B": "Standard", "C": "Basic"}
        ...             }
        ...         }
        ...     ],
        ...     group_by=["category", "region"],
        ... )
        Dataset saved successfully to path: datasets/my-org/sales
    """
    if df is not None and not isinstance(df, DataFrame):
        raise ValueError("df must be a PandasAI DataFrame")

    org_name, dataset_name = get_validated_dataset_path(path)
    underscore_dataset_name = transform_dash_to_underscore(dataset_name)
    dataset_directory = str(os.path.join(org_name, dataset_name))

    schema_path = os.path.join(dataset_directory, "schema.yaml")
    parquet_file_path = os.path.join(dataset_directory, "data.parquet")

    file_manager = config.get().file_manager
    # Check if dataset already exists
    if file_manager.exists(dataset_directory) and file_manager.exists(schema_path):
        raise ValueError(f"Dataset already exists at path: {path}")

    file_manager.mkdir(dataset_directory)

    if df is None and source is None and not view:
        raise InvalidConfigError(
            "Please provide either a DataFrame, a Source or a View"
        )

    # Parse transformations if provided
    parsed_transformations = (
        [Transformation(**t) for t in transformations] if transformations else None
    )
    parsed_columns = [Column(**column) for column in columns] if columns else None

    if df is not None:
        schema = df.schema
        schema.name = underscore_dataset_name
        schema.transformations = parsed_transformations
        if (
            parsed_columns
        ):  # if no columns are passed it automatically parse the columns from the df
            schema.columns = parsed_columns
        if group_by is not None:
            schema.group_by = group_by
        SemanticLayerSchema.model_validate(schema)
        parquet_file_path_abs_path = file_manager.abs_path(parquet_file_path)
        df.to_parquet(parquet_file_path_abs_path, index=False)
    elif view:
        _relation = [Relation(**relation) for relation in relations or ()]
        schema: SemanticLayerSchema = SemanticLayerSchema(
            name=underscore_dataset_name,
            relations=_relation,
            view=True,
            columns=parsed_columns,
            group_by=group_by,
            transformations=parsed_transformations,
        )
    elif source.get("table"):
        schema: SemanticLayerSchema = SemanticLayerSchema(
            name=underscore_dataset_name,
            source=Source(**source),
            columns=parsed_columns,
            group_by=group_by,
            transformations=parsed_transformations,
        )

    schema.description = description or schema.description

    file_manager.write(schema_path, schema.to_yaml())

    print(f"Dataset saved successfully to path: {dataset_directory}")

    schema.name = sanitize_sql_table_name(schema.name)
    loader = DatasetLoader.create_loader_from_schema(schema, path)
    return loader.load()


# Global variable to store the current agent
_current_agent = None

config = ConfigManager()

api_key = APIKeyManager()

skills = SkillsManager()


def chat(query: str, *dataframes: DataFrame, sandbox: Optional[Sandbox] = None):
    """
    Start a new chat interaction with the assistant on Dataframe(s).

    Args:
        query (str): The query to run against the dataframes.
        *dataframes: Variable number of dataframes to query.
        sandbox (Sandbox, optional): The sandbox to execute code securely.

    Returns:
        The result of the query.
    """
    global _current_agent
    if not dataframes:
        raise ValueError("At least one dataframe must be provided.")

    _current_agent = Agent(list(dataframes), sandbox=sandbox)
    return _current_agent.chat(query)


def follow_up(query: str):
    """
    Continue the existing chat interaction with the assistant on Dataframe(s).

    Args:
        query (str): The follow-up query to run.

    Returns:
        The result of the query.
    """
    global _current_agent

    if _current_agent is None:
        raise ValueError(
            "No existing conversation. Please use chat() to start a new conversation."
        )

    return _current_agent.follow_up(query)


def load(dataset_path: str) -> DataFrame:
    """
    Load data based on the provided dataset path.

    Args:
        dataset_path (str): Path in the format 'organization/dataset_name'.

    Returns:
        DataFrame: A new PandasAI DataFrame instance with loaded data.
    """

    # Validate the dataset path
    get_validated_dataset_path(dataset_path)

    dataset_full_path = os.path.join(find_project_root(), "datasets", dataset_path)

    local_dataset_exists = os.path.exists(dataset_full_path)

    if not local_dataset_exists:
        raise DatasetNotFound("Dataset not found!")

    loader = DatasetLoader.create_loader_from_path(dataset_path)
    df = loader.load()

    message = (
        "Dataset loaded successfully."
        if local_dataset_exists
        else "Dataset fetched successfully from the remote server."
    )
    # Printed to display info to the user
    print(message)

    return df


def read_csv(filepath: Union[str, BytesIO]) -> DataFrame:
    data = pd.read_csv(filepath)
    table = get_table_name_from_path(filepath)
    return DataFrame(data, _table_name=table)


def read_excel(
    filepath: Union[str, BytesIO],
    sheet_name: Union[str, int, list[Union[str, int]], None] = 0,
) -> dict[Hashable, DataFrame] | DataFrame:
    data = pd.read_excel(filepath, sheet_name=sheet_name)

    if isinstance(data, pd.DataFrame):
        table = get_table_name_from_path(filepath)
        return DataFrame(data, _table_name=table)

    return {
        k: DataFrame(
            v,
            _table_name=sanitize_sql_table_name_lowercase(
                f"{get_table_name_from_path(filepath)}_{k}"
            ),
        )
        for k, v in data.items()
    }


__all__ = [
    "Agent",
    "DataFrame",
    "VirtualDataFrame",
    "pandas",
    "chat",
    "follow_up",
    "load",
    "skill",
    # Deprecated
    "SmartDataframe",
    "SmartDatalake",
]
