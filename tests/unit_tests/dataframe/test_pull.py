import os
from io import BytesIO
from unittest.mock import Mock, PropertyMock, mock_open, patch
from zipfile import ZipFile

import pandas as pd
import pytest

from pandasai.config import ConfigManager
from pandasai.data_loader.semantic_layer_schema import (
    Column,
    SemanticLayerSchema,
    Source,
)
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import DatasetNotFound, PandasAIApiKeyError
from pandasai.helpers.path import find_project_root


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("PANDABI_API_KEY", "test_api_key")


@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@pytest.fixture
def mock_zip_content():
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("test.csv", "col1,col2\n1,a\n2,b\n3,c")
    return zip_buffer.getvalue()


@pytest.fixture
def mock_schema():
    return SemanticLayerSchema(
        name="test_schema",
        source=Source(type="parquet", path="data.parquet", table="test_table"),
        columns=[
            Column(name="col1", type="integer"),
            Column(name="col2", type="string"),
        ],
    )


def test_pull_success(mock_env, sample_df, mock_zip_content, mock_schema, tmp_path):
    with patch("pandasai.dataframe.base.get_PandasAI_session") as mock_session, patch(
        "pandasai.helpers.path.find_project_root"
    ) as mock_root, patch(
        "pandasai.DatasetLoader.create_loader_from_path"
    ) as mock_loader, patch("builtins.open", mock_open()) as mock_file:
        # Setup mocks
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_zip_content
        mock_session.return_value.get.return_value = mock_response
        mock_root.return_value = str(tmp_path)

        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = DataFrame(
            sample_df, schema=mock_schema
        )
        mock_loader.return_value = mock_loader_instance

        # Create DataFrame instance and call pull
        df = DataFrame(sample_df, path="test/path", schema=mock_schema)
        df.pull()

        # Verify API call
        mock_session.return_value.get.assert_called_once_with(
            "/datasets/pull",
            headers={
                "accept": "application/json",
                "x-authorization": "Bearer test_api_key",
            },
            params={"path": "test/path"},
        )

        # Verify file operations
        assert mock_file.call_count > 0


def test_pull_missing_api_key(sample_df, mock_schema):
    with patch("os.environ.get") as mock_env_get:
        mock_env_get.return_value = None
        with pytest.raises(PandasAIApiKeyError):
            df = DataFrame(sample_df, path="test/path", schema=mock_schema)
            df.pull()


def test_pull_api_error(mock_env, sample_df, mock_schema):
    with patch("pandasai.dataframe.base.get_PandasAI_session") as mock_session:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.return_value.get.return_value = mock_response

        df = DataFrame(sample_df, path="test/path", schema=mock_schema)
        with pytest.raises(DatasetNotFound, match="Remote dataset not found to pull!"):
            df.pull()


def test_pull_file_exists(mock_env, sample_df, mock_zip_content, mock_schema, tmp_path):
    with patch("pandasai.dataframe.base.get_PandasAI_session") as mock_session, patch(
        "pandasai.helpers.path.find_project_root"
    ) as mock_root, patch(
        "pandasai.DatasetLoader.create_loader_from_path"
    ) as mock_loader, patch("builtins.open", mock_open()) as mock_file, patch(
        "os.path.exists"
    ) as mock_exists, patch("os.makedirs") as mock_makedirs:
        # Setup mocks
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_zip_content
        mock_session.return_value.get.return_value = mock_response
        mock_root.return_value = str(tmp_path)
        mock_exists.return_value = True

        # Setup FileManager mock
        mock_file_manager = Mock()
        mock_file_manager.exists.return_value = True
        mock_file_manager.base_path = os.path.join(str(tmp_path), "datasets")
        mock_config = Mock()
        mock_config.file_manager = mock_file_manager
        with patch.object(ConfigManager, "get", return_value=mock_config):
            mock_loader_instance = Mock()
            mock_loader_instance.load.return_value = DataFrame(
                sample_df, schema=mock_schema
            )
            mock_loader.return_value = mock_loader_instance

            # Create DataFrame instance and call pull
            df = DataFrame(sample_df, path="test/path", schema=mock_schema)
            df.pull()

            # Verify directory creation
            mock_file_manager.mkdir.assert_called_with(
                os.path.dirname("test/path/test.csv")
            )
