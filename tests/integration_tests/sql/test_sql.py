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
    mock_sql_df,
    root_dir,
    set_fake_llm_output,
)


@pytest.fixture(scope="session")
def sql_dataset_slug():
    connection = {
        "host": "example.amazonaws.com",
        "port": 5432,
        "user": "user",
        "password": "password",
        "database": "db",
    }

    source = {"type": "postgres", "connection": connection, "table": "parents"}
    columns = [
        {
            "name": "id",
        },
        {
            "name": "name",
        },
    ]
    _id = uuid.uuid4()
    dataset_org = f"integration-test-organization-{_id}"
    dataset_path = f"testing-dataset-{_id}"
    dataset_slug = f"{dataset_org}/{dataset_path}"
    pai.create(
        dataset_slug,
        source=source,
        description="integration test postgres dataset",
        columns=columns,
    )
    yield dataset_slug
    shutil.rmtree(f"{root_dir}/datasets/{dataset_org}")


def test_slug_fixture(sql_dataset_slug):
    assert re.match(
        r"integration-test-organization-[0-9a-f-]+/testing-dataset-[0-9a-f-]+",
        sql_dataset_slug,
    )


def test_sql_files(sql_dataset_slug, root_path):
    schema_path = f"{root_path}/datasets/{sql_dataset_slug}/schema.yaml"

    assert os.path.exists(schema_path)


def test_sql_load(sql_dataset_slug):
    dataset = pai.load(sql_dataset_slug)

    compare_sorted_dataframe(dataset.head(), mock_sql_df, "column 1")


def test_sql_chat(sql_dataset_slug):
    dataset = pai.load(sql_dataset_slug)

    set_fake_llm_output(
        output=f"""import pandas as pd
sql_query = 'SELECT * FROM {dataset.schema.name}'
df = execute_sql_query(sql_query)
result = {{'type': 'dataframe', 'value': df}}"""
    )

    result = dataset.chat("Give me all the dataset")
    compare_sorted_dataframe(result.value, mock_sql_df, "column 1")


def test_sql_push(sql_dataset_slug, mock_pandasai_push, capsys):
    dataset = pai.load(sql_dataset_slug)
    dataset.push()
    captured = capsys.readouterr()
    assert "Your dataset was successfully pushed to the remote server!" in captured.out
    assert (
        "https://app.pandabi.ai/datasets/integration-test-organization" in captured.out
    )


def test_sql_pull(sql_dataset_slug, mock_dataset_pull):
    dataset = pai.load(sql_dataset_slug)
    dataset.pull()
