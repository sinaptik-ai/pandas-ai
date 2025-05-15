from unittest.mock import MagicMock, Mock, mock_open, patch

import pandas as pd
import pytest

import pandasai
from pandasai.agent import Agent
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import PandasAIApiKeyError


class TestDataFrame:
    @pytest.fixture(autouse=True)
    def reset_current_agent(self):
        pandasai._current_agent = None
        yield
        pandasai._current_agent = None

    def test_dataframe_initialization(self, sample_dict_data, sample_df):
        assert isinstance(sample_df, DataFrame)
        assert isinstance(sample_df, pd.DataFrame)
        assert sample_df.equals(pd.DataFrame(sample_dict_data))

    def test_dataframe_operations(self, sample_df):
        assert len(sample_df) == 3
        assert list(sample_df.columns) == ["A", "B"]
        assert sample_df["A"].mean() == 2

    @patch("pandasai.agent.Agent")
    @patch("os.environ")
    def test_chat_creates_agent(self, mock_env, mock_agent, sample_dict_data):
        sample_df = DataFrame(sample_dict_data)
        mock_env.return_value = {"PANDABI_API_URL": "localhost:8000"}
        sample_df.chat("Test query")
        mock_agent.assert_called_once_with([sample_df], sandbox=None)

    @patch("pandasai.agent.Agent")
    @patch("os.environ")
    def test_chat_creates_agent_with_sandbox(
        self, mock_env, mock_agent, sample_dict_data
    ):
        sandbox = MagicMock()
        sample_df = DataFrame(sample_dict_data)
        mock_env.return_value = {"PANDABI_API_URL": "localhost:8000"}
        sample_df.chat("Test query", sandbox=sandbox)
        mock_agent.assert_called_once_with([sample_df], sandbox=sandbox)

    @patch("pandasai.Agent")
    def test_chat_reuses_existing_agent(self, sample_df):
        mock_agent = Mock(spec=Agent)
        sample_df._agent = mock_agent

        sample_df.chat("First query")
        assert sample_df._agent is not None
        initial_agent = sample_df._agent
        sample_df.chat("Second query")
        assert sample_df._agent is initial_agent

    def test_follow_up_without_chat_raises_error(self, sample_df):
        with pytest.raises(ValueError, match="No existing conversation"):
            sample_df.follow_up("Follow-up query")

    def test_follow_up_after_chat(self, sample_df):
        mock_agent = Mock(spec=Agent)
        sample_df._agent = mock_agent

        sample_df.follow_up("Follow-up query")
        assert mock_agent.follow_up.call_count == 1

    def test_chat_method(self, sample_df):
        mock_agent = Mock(spec=Agent)
        sample_df._agent = mock_agent

        sample_df.chat("Test question")

        assert sample_df._agent is not None
        assert mock_agent.chat.call_count == 1

    def test_column_hash(self, sample_df):
        assert hasattr(sample_df, "column_hash")
        assert isinstance(sample_df.column_hash, str)
        assert len(sample_df.column_hash) == 32  # MD5 hash length

    @patch("pandasai.dataframe.base.get_PandasAI_session")
    @patch("pandasai.helpers.filemanager.os.path.exists")
    @patch("pandasai.helpers.filemanager.open", new_callable=mock_open)
    @patch("pandasai.dataframe.base.os.environ")
    @patch("pandasai.helpers.path.find_project_root")
    def test_push_successful(
        self,
        mock_find_project_root,
        mock_environ,
        mock_open,
        mock_path_exists,
        mock_get_session,
        sample_df,
    ):
        # Mock environment variable
        mock_environ.get.return_value = "fake_api_key"

        # Mock project root
        mock_find_project_root.return_value = "/fake/project/root"

        # Mock file paths
        mock_path_exists.side_effect = (
            lambda x: "data.parquet" in x
        )  # Only data.parquet exists

        # Mock session and POST request
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        sample_df.path = "test/test"
        sample_df.push()

        # Verify POST request
        mock_session.post.assert_called_once_with(
            "/datasets/push",
            files=[
                (
                    "files",
                    ("schema.yaml", "", "application/x-yaml"),
                ),
                (
                    "files",
                    (
                        "data.parquet",
                        "",
                        "application/octet-stream",
                    ),
                ),
            ],
            params={
                "path": sample_df.path,
                "description": sample_df.schema.description,
                "name": sample_df.schema.name,
            },
            headers={
                "accept": "application/json",
                "x-authorization": "Bearer fake_api_key",
            },
        )

    def test_push_raises_error_if_path_is_none(self, sample_df):
        # Call the method and assert the exception
        with pytest.raises(
            ValueError,
            match="Please save the dataset before pushing to the remote server.",
        ) as context:
            sample_df.push()

    @patch("pandasai.dataframe.base.os.environ")
    def test_push_raises_error_if_api_key_is_missing(self, mock_environ, sample_df):
        # Mock environment variable as missing
        mock_environ.get.return_value = None

        # Call the method and assert the exception
        with pytest.raises(PandasAIApiKeyError):
            sample_df.path = "test/test"
            sample_df.push()

    @patch("pandasai.helpers.filemanager.os.path.exists")
    @patch("pandasai.helpers.filemanager.open", new_callable=mock_open)
    @patch("pandasai.dataframe.base.get_PandasAI_session")
    @patch("pandasai.dataframe.base.os.environ")
    def test_push_closes_files_on_completion(
        self,
        mock_environ,
        mock_get_session,
        mock_open,
        mock_path_exists,
        sample_df: DataFrame,
    ):
        # Mock environment variable
        mock_environ.get.return_value = "fake_api_key"
        # Mock file existence
        mock_path_exists.return_value = True

        # Mock session
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Call the method
        sample_df.path = "test/test"
        sample_df.push()
