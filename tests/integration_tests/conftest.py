import os
from io import BytesIO
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pandasai as pai
from pandasai.data_loader.semantic_layer_schema import SemanticLayerSchema, Source
from pandasai.dataframe.base import DataFrame
from pandasai.helpers.path import find_project_root
from pandasai.llm.fake import FakeLLM

root_dir = find_project_root()


@pytest.fixture
def mock_pandasai_push():
    """Fixture to mock the HTTP POST request in pandasai.helpers.session."""
    with patch("pandasai.helpers.session.requests.request") as mock_request:
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Dataset pushed successfully"}
        mock_request.return_value = mock_response
        yield mock_request


@pytest.fixture
def mock_dataset_pull():
    """Fixture to mock the GET request, endpoint URL, and file operations for dataset pull."""

    schema = SemanticLayerSchema(
        name="test_schema", source=Source(type="parquet", path="data.parquet")
    )

    df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    table = pa.Table.from_pandas(df)

    # Write to an in-memory buffer
    parquet_buffer = BytesIO()
    pq.write_table(table, parquet_buffer)
    parquet_buffer.seek(0)
    parquet_bytes = parquet_buffer.getvalue()

    # Create a fake ZIP file in memory
    fake_zip_bytes = BytesIO()
    with ZipFile(fake_zip_bytes, "w") as fake_zip:
        fake_zip.writestr("data.parquet", parquet_bytes)
        fake_zip.writestr("schema.yaml", schema.to_yaml())
    fake_zip_bytes.seek(0)

    # We need to patch the session.get method to return a response-like object
    with patch("pandasai.dataframe.base.get_pandaai_session") as mock_session_getter:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_zip_bytes.read()
        mock_session_getter.return_value.get.return_value = mock_response

        yield mock_session_getter


@pytest.fixture
def root_path():
    return root_dir


@pytest.fixture(autouse=True)
def clear_os_environ(monkeypatch):
    # Clear all environment variables
    for var in list(os.environ.keys()):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("PANDABI_API_KEY", "test_api_key")
    monkeypatch.setenv("PANDABI_API_URL", "test_api_url")


mock_sql_df = DataFrame(
    {
        "column 1": [1, 2, 3, 4, 5, 6],
        "column 2": ["a", "b", "c", "d", "e", "f"],
        "column 3": [1, 2, 3, 4, 5, 6],
        "column 4": ["a", "b", "c", "d", "e", "f"],
    }
)


@pytest.fixture(autouse=True)
def mock_sql_load_function():
    with patch(
        "pandasai.data_loader.sql_loader.SQLDatasetLoader._get_loader_function"
    ) as mock_loader_function:
        mocked_exec_function = MagicMock()

        mocked_exec_function.return_value = mock_sql_df
        mock_loader_function.return_value = mocked_exec_function
        yield mock_loader_function


def set_fake_llm_output(output: str):
    fake_llm = FakeLLM(output=output)
    pai.config.set({"llm": fake_llm})


def compare_sorted_dataframe(df1: pd.DataFrame, df2: pd.DataFrame, column: str):
    pd.testing.assert_frame_equal(
        df1.sort_values(by=column).reset_index(drop=True),
        df2.sort_values(by=column).reset_index(drop=True),
        check_like=True,
    )
