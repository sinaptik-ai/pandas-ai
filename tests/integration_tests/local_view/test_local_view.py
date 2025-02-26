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
        "user_id": [1, 2, 3, 4, 5],
        "username": ["alice", "bob", "carol", "dave", "eve"],
        "user_age": [25, 30, 22, 35, 28],
        "detail_id": [101, 102, 103, 104, 105],
        "email_address": [
            "alice@example.com",
            "bob@example.com",
            "carol@example.com",
            "dave@example.com",
            "eve@example.com",
        ],
        "country": ["USA", "UK", "Canada", "Germany", "France"],
    }
)


@pytest.fixture(scope="session")
def local_view_dataset_slug():
    users_dataframe = DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "username": ["alice", "bob", "carol", "dave", "eve", "frank"],
            "age": [25, 30, 22, 35, 28, 40],
        }
    )

    users_details_dataframe = DataFrame(
        {
            "detail_id": [101, 102, 103, 104, 105, 106],  # Primary Key
            "user_id": [1, 2, 3, 4, 5, 6],  # Foreign Key (refers to df1.user_id)
            "email": [
                "alice@example.com",
                "bob@example.com",
                "carol@example.com",
                "dave@example.com",
                "eve@example.com",
                "frank@example.com",
            ],
            "country": ["USA", "UK", "Canada", "Germany", "France", "Australia"],
        }
    )

    view_id = uuid.uuid4()
    dataset_org = f"integration-test-organization-{view_id}"

    view_path = f"testing-dataset-{view_id}"
    view_slug = f"{dataset_org}/{view_path}"

    users_path = "users"
    users_slug = f"{dataset_org}/{users_path}"

    users_details_path = "users-details"
    users_details_slug = f"{dataset_org}/{users_details_path}"

    pai.create(f"{users_slug}", users_dataframe, description="users dataframe")
    pai.create(users_details_slug, users_details_dataframe, description="heart")

    view_columns = [
        {"name": "users.user_id", "alias": "user_id"},
        {"name": "users.username", "alias": "username"},
        {"name": "users.age", "alias": "user_age"},
        {"name": "users_details.detail_id", "alias": "detail_id"},
        {"name": "users_details.email", "alias": "email_address"},
        {"name": "users_details.country", "alias": "country"},
    ]

    view_relations = [{"from": "users.user_id", "to": "users_details.user_id"}]

    pai.create(
        view_slug,
        description="health-diabetes-combined",
        view=True,
        columns=view_columns,
        relations=view_relations,
    )
    yield view_slug

    shutil.rmtree(f"{root_dir}/datasets/{dataset_org}")


def test_slug_fixture(local_view_dataset_slug):
    assert re.match(
        r"integration-test-organization-[0-9a-f-]+/testing-dataset-[0-9a-f-]+",
        local_view_dataset_slug,
    )


def test_local_view_files(local_view_dataset_slug, root_path):
    org = local_view_dataset_slug.split("/")[0]

    view_schema_path = f"{root_path}/datasets/{local_view_dataset_slug}/schema.yaml"
    users_schema_path = f"{root_path}/datasets/{org}/users/schema.yaml"
    users_data_path = f"{root_path}/datasets/{org}/users/data.parquet"

    users_details_schema_path = f"{root_path}/datasets/{org}/users-details/schema.yaml"
    users_details_data_path = f"{root_path}/datasets/{org}/users-details/data.parquet"

    assert os.path.exists(view_schema_path)
    assert os.path.exists(users_schema_path)
    assert os.path.exists(users_data_path)
    assert os.path.exists(users_details_schema_path)
    assert os.path.exists(users_details_data_path)


def test_local_view_load(local_view_dataset_slug):
    dataset = pai.load(local_view_dataset_slug)

    compare_sorted_dataframe(dataset.head(), expected_df, "user_id")


def test_local_view_chat(local_view_dataset_slug):
    dataset = pai.load(local_view_dataset_slug)

    set_fake_llm_output(
        output=f"""import pandas as pd
sql_query = 'SELECT * FROM {dataset.schema.name}'
df = execute_sql_query(sql_query)
result = {{'type': 'dataframe', 'value': df}}"""
    )

    result = dataset.chat("Give me all the dataset")
    compare_sorted_dataframe(result.value.head(), expected_df, "user_id")


def test_local_view_push(local_view_dataset_slug, mock_pandasai_push, capsys):
    dataset = pai.load(local_view_dataset_slug)
    dataset.push()
    captured = capsys.readouterr()
    assert "Your dataset was successfully pushed to the remote server!" in captured.out
    assert (
        "https://app.pandabi.ai/datasets/integration-test-organization" in captured.out
    )


def test_local_view_pull(local_view_dataset_slug, mock_dataset_pull):
    dataset = pai.load(local_view_dataset_slug)
    dataset.pull()
