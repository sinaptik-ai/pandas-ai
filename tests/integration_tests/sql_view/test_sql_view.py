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
def sql_view_dataset_slug():
    connection = {
        "host": "example.amazonaws.com",
        "port": 5432,
        "user": "user",
        "password": "password",
        "database": "db",
    }
    parents_source = {
        "type": "postgres",
        "connection": connection,
        "table": "us_parents",
    }
    parents_columns = [
        {
            "name": "id",
        },
        {
            "name": "name",
        },
    ]
    children_source = {
        "type": "postgres",
        "connection": connection,
        "table": "us_children",
    }
    children_columns = [
        {
            "name": "id",
        },
        {
            "name": "name",
        },
        {"name": "parent_id"},
    ]
    view_columns = [
        {"name": "us_parents.id"},
        {"name": "us_parents.name"},
        {"name": "us_children.id"},
        {"name": "us_children.name"},
    ]

    view_relations = [{"from": "us_parents.id", "to": "us_children.parent_id"}]

    view_id = uuid.uuid4()
    dataset_org = f"integration-test-organization-{view_id}"

    view_path = f"testing-dataset-{view_id}"
    view_slug = f"{dataset_org}/{view_path}"

    parents_path = "us-parents"
    parents_slug = f"{dataset_org}/{parents_path}"

    children_path = "us-children"
    children_slug = f"{dataset_org}/{children_path}"

    pai.create(
        parents_slug,
        source=parents_source,
        columns=parents_columns,
        description="parents dataset",
    )
    pai.create(
        children_slug,
        source=children_source,
        columns=children_columns,
        description="children dataset",
    )

    pai.create(
        view_slug,
        description="sql view",
        view=True,
        columns=view_columns,
        relations=view_relations,
    )
    yield view_slug

    shutil.rmtree(f"{root_dir}/datasets/{dataset_org}")


def test_slug_fixture(sql_view_dataset_slug):
    assert re.match(
        r"integration-test-organization-[0-9a-f-]+/testing-dataset-[0-9a-f-]+",
        sql_view_dataset_slug,
    )


def test_sql_view_files(sql_view_dataset_slug, root_path):
    org = sql_view_dataset_slug.split("/")[0]

    view_schema_path = f"{root_path}/datasets/{sql_view_dataset_slug}/schema.yaml"
    us_parents_schema_path = f"{root_path}/datasets/{org}/us-parents/schema.yaml"
    us_children_schema_path = f"{root_path}/datasets/{org}/us-children/schema.yaml"

    assert os.path.exists(view_schema_path)
    assert os.path.exists(us_parents_schema_path)
    assert os.path.exists(us_children_schema_path)


def test_sql_view_load(sql_view_dataset_slug):
    dataset = pai.load(sql_view_dataset_slug)

    compare_sorted_dataframe(dataset.head(), mock_sql_df, "column 1")


def test_sql_view_chat(sql_view_dataset_slug):
    dataset = pai.load(sql_view_dataset_slug)

    set_fake_llm_output(
        output=f"""import pandas as pd
sql_query = 'SELECT * FROM {dataset.schema.name}'
df = execute_sql_query(sql_query)
result = {{'type': 'dataframe', 'value': df}}"""
    )

    result = dataset.chat("Give me all the dataset")
    compare_sorted_dataframe(result.value, mock_sql_df, "column 1")


def test_sql_view_push(sql_view_dataset_slug, mock_pandasai_push, capsys):
    dataset = pai.load(sql_view_dataset_slug)
    dataset.push()
    captured = capsys.readouterr()
    assert "Your dataset was successfully pushed to the remote server!" in captured.out
    assert (
        "https://app.pandabi.ai/datasets/integration-test-organization" in captured.out
    )


def test_sql_view_pull(sql_view_dataset_slug, mock_dataset_pull):
    dataset = pai.load(sql_view_dataset_slug)
    dataset.pull()
