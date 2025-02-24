import os
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from pandasai import ConfigManager
from pandasai.data_loader.loader import DatasetLoader
from pandasai.data_loader.semantic_layer_schema import SemanticLayerSchema
from pandasai.data_loader.sql_loader import SQLDatasetLoader
from pandasai.dataframe.base import DataFrame
from pandasai.helpers.path import find_project_root
from pandasai.llm.fake import FakeLLM
from pandasai.query_builders.sql_query_builder import SqlQueryBuilder


@pytest.fixture
def sample_dict_data():
    return {"A": [1, 2, 3], "B": [4, 5, 6]}


@pytest.fixture
def sample_df(sample_dict_data):
    return DataFrame(sample_dict_data)


@pytest.fixture
def sample_dataframes():
    df1 = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    df2 = DataFrame({"X": [10, 20, 30], "Y": ["x", "y", "z"]})
    return [df1, df2]


@pytest.fixture
def raw_sample_schema():
    return {
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
        "source": {"type": "csv", "path": "users.csv", "table": "users"},
    }


@pytest.fixture
def raw_mysql_schema():
    return {
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


@pytest.fixture
def raw_mysql_view_schema():
    return {
        "name": "parent_children",
        "columns": [
            {"name": "parents.id"},
            {"name": "parents.name"},
            {"name": "children.name"},
        ],
        "relations": [{"from": "parents.id", "to": "children.id"}],
        "view": "true",
    }


@pytest.fixture
def sample_schema(raw_sample_schema):
    return SemanticLayerSchema(**raw_sample_schema)


@pytest.fixture
def mysql_schema(raw_mysql_schema):
    return SemanticLayerSchema(**raw_mysql_schema)


@pytest.fixture
def mock_view_loader_instance_parents(sample_df):
    """Fixture to mock DatasetLoader and its methods."""
    # Mock the create_loader_from_path method
    mock_loader_instance = MagicMock(spec=SQLDatasetLoader)
    mock_loader_instance.load.return_value = sample_df
    schema = SemanticLayerSchema(
        **{
            "name": "parents",
            "source": {
                "type": "mysql",
                "connection": {
                    "host": "localhost",
                    "port": 3306,
                    "database": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                },
                "table": "parents",
            },
        }
    )
    mock_query_builder = SqlQueryBuilder(schema=schema)
    mock_loader_instance.query_builder = mock_query_builder
    mock_loader_instance.schema = schema
    yield mock_loader_instance


@pytest.fixture
def mock_view_loader_instance_children(sample_df):
    """Fixture to mock DatasetLoader and its methods."""
    # Mock the create_loader_from_path method
    mock_loader_instance = MagicMock(spec=SQLDatasetLoader)
    mock_loader_instance.load.return_value = sample_df
    schema = SemanticLayerSchema(
        **{
            "name": "children",
            "source": {
                "type": "mysql",
                "connection": {
                    "host": "localhost",
                    "port": 3306,
                    "database": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                },
                "table": "children",
            },
        }
    )
    mock_query_builder = SqlQueryBuilder(schema=schema)
    mock_loader_instance.query_builder = mock_query_builder
    mock_loader_instance.schema = schema
    yield mock_loader_instance


@pytest.fixture
def mysql_view_schema(raw_mysql_view_schema):
    return SemanticLayerSchema(**raw_mysql_view_schema)


@pytest.fixture
def mysql_view_dependencies_dict(
    mock_view_loader_instance_parents, mock_view_loader_instance_children
) -> dict[str, MagicMock]:
    return {
        "parents": mock_view_loader_instance_parents,
        "children": mock_view_loader_instance_children,
    }


@pytest.fixture(scope="session")
def mock_json_load():
    mock = MagicMock()

    with patch("json.load", mock):
        yield mock


def pytest_terminal_summary(terminalreporter, exitstatus):
    scores_file = Path(find_project_root()) / "test_agent_llm_judge.txt"

    if os.path.exists(scores_file):
        with open(scores_file, "r") as file:
            score_line = file.readline().strip()

            # Ensure the line is a valid number
            if score_line.replace(".", "", 1).isdigit():
                avg_score = float(score_line)
                terminalreporter.write(f"\n--- Evaluation Score Summary ---\n")
                terminalreporter.write(f"Average Score: {avg_score:.2f}\n")

        os.remove(scores_file)


@pytest.fixture
def mock_loader_instance(sample_df):
    """Fixture to mock DatasetLoader and its methods."""
    with patch.object(
        DatasetLoader, "create_loader_from_path"
    ) as mock_create_loader, patch.object(
        DatasetLoader, "create_loader_from_schema"
    ) as mock_create_loader_from_schema:
        # Mock the create_loader_from_path method
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_df
        mock_create_loader.return_value = mock_loader_instance
        mock_create_loader_from_schema.return_value = mock_loader_instance

        yield mock_loader_instance


@pytest.fixture
def mock_file_manager():
    """Fixture to mock FileManager and its methods."""
    with patch.object(ConfigManager, "get") as mock_config_get:
        # Create a mock FileManager
        mock_file_manager = MagicMock()
        mock_file_manager.exists.return_value = False
        mock_config_get.return_value.file_manager = mock_file_manager
        yield mock_file_manager


@pytest.fixture
def llm(output: Optional[str] = None) -> FakeLLM:
    return FakeLLM(output=output)
