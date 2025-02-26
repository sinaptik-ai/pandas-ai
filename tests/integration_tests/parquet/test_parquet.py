import os.path
import re
import shutil
import uuid

import pandas as pd
import pytest

import pandasai as pai
from pandasai import DataFrame
from tests.integration_tests.conftest import (
    compare_sorted_dataframe,
    root_dir,
    set_fake_llm_output,
)

expected_df = pd.DataFrame(
    {
        "column 1": [1, 2, 3, 4, 5, 6],
        "column 2": ["a", "b", "c", "d", "e", "f"],
        "column 3": [1, 2, 3, 4, 5, 6],
        "column 4": ["a", "b", "c", "d", "e", "f"],
    }
)


@pytest.fixture(scope="session")
def parquet_dataset_slug():
    # Setup code
    df = DataFrame(expected_df)
    _id = uuid.uuid4()
    dataset_org = f"integration-test-organization-{_id}"
    dataset_path = f"testing-dataset-{_id}"
    dataset_slug = f"{dataset_org}/{dataset_path}"
    pai.create(dataset_slug, df, description="integration test local dataset")
    yield dataset_slug
    shutil.rmtree(f"{root_dir}/datasets/{dataset_org}")


def test_slug_fixture(parquet_dataset_slug):
    assert re.match(
        r"integration-test-organization-[0-9a-f-]+/testing-dataset-[0-9a-f-]+",
        parquet_dataset_slug,
    )


def test_parquet_files(parquet_dataset_slug, root_path):
    parquet_path = f"{root_path}/datasets/{parquet_dataset_slug}/data.parquet"
    schema_path = f"{root_path}/datasets/{parquet_dataset_slug}/schema.yaml"

    assert os.path.exists(parquet_path)
    assert os.path.exists(schema_path)


def test_parquet_load(parquet_dataset_slug):
    dataset = pai.load(parquet_dataset_slug)

    compare_sorted_dataframe(dataset, expected_df, "column 1")


def test_parquet_chat(parquet_dataset_slug):
    dataset = pai.load(parquet_dataset_slug)

    set_fake_llm_output(
        output=f"""import pandas as pd
sql_query = 'SELECT * FROM {dataset.schema.name}'
df = execute_sql_query(sql_query)
result = {{'type': 'dataframe', 'value': df}}"""
    )

    result = dataset.chat("Give me all the dataset")
    compare_sorted_dataframe(result.value, expected_df, "column 1")


def test_parquet_push(parquet_dataset_slug, mock_pandasai_push, capsys):
    dataset = pai.load(parquet_dataset_slug)
    dataset.push()
    captured = capsys.readouterr()
    assert "Your dataset was successfully pushed to the remote server!" in captured.out
    assert (
        "https://app.pandabi.ai/datasets/integration-test-organization" in captured.out
    )


def test_parquet_pull(parquet_dataset_slug, mock_dataset_pull):
    dataset = pai.load(parquet_dataset_slug)
    dataset.pull()
