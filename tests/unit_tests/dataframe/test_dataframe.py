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
