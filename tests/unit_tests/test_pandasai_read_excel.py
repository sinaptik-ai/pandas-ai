import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import patch, MagicMock

import pandasai


class TestReadExcel:
    """Test suite for the read_excel function."""

    @pytest.fixture
    def mock_dataframe_class(self):
        """Mock DataFrame class that mimics your custom DataFrame."""
        mock_df = MagicMock()
        mock_df.return_value = MagicMock()
        return mock_df

    @pytest.fixture
    def sample_pandas_df(self):
        """Sample pandas DataFrame for testing."""
        return pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

    @pytest.fixture
    def sample_multi_sheet_data(self, sample_pandas_df):
        """Sample multi-sheet data structure."""
        return {
            'Sheet1': sample_pandas_df,
            'Sheet2': pd.DataFrame({'col3': [4, 5, 6]}),
            'Sheet3': pd.DataFrame({'col4': ['x', 'y', 'z']})
        }

    @patch('pandas.read_excel')
    def test_read_excel_single_sheet_string_filepath(
        self, mock_pd_read_excel, sample_pandas_df
    ):
        """Test reading Excel with single sheet and string filepath."""
        # Setup
        filepath = "/path/to/file.xlsx"
        mock_pd_read_excel.return_value = sample_pandas_df

        pandasai.read_excel(filepath)
        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)

    @patch('pandas.read_excel')
    def test_read_excel_single_sheet_bytesio_filepath(
        self, mock_pd_read_excel, sample_pandas_df
    ):
        """Test reading Excel with single sheet and BytesIO filepath."""
        # Setup
        filepath = BytesIO(b"fake excel content")
        mock_pd_read_excel.return_value = sample_pandas_df

        pandasai.read_excel(filepath)
        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)

    @patch('pandas.read_excel')
    def test_read_excel_multi_sheet_no_sheet_name_string_filepath(
        self, mock_pd_read_excel, sample_multi_sheet_data
    ):
        """Test reading Excel with multiple sheets, no sheet_name specified, string filepath."""
        # Setup
        filepath = "/path/to/file.xlsx"
        mock_pd_read_excel.return_value = sample_multi_sheet_data

        result = pandasai.read_excel(filepath)

        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)
        assert isinstance(result, dict)
        assert len(result) == 3

        for sheet_name, df in result.items():
            assert sheet_name in sample_multi_sheet_data
            assert isinstance(df, pd.DataFrame)
            assert result[sheet_name].equals(sample_multi_sheet_data[sheet_name])

    @patch('pandas.read_excel')
    def test_read_excel_multi_sheet_no_sheet_name_bytesio_filepath(
        self, mock_pd_read_excel, sample_multi_sheet_data
    ):
        """Test reading Excel with multiple sheets, no sheet_name specified, BytesIO filepath."""
        # Setup
        filepath = BytesIO(b"fake excel content")
        mock_pd_read_excel.return_value = sample_multi_sheet_data

        # Execute
        result = pandasai.read_excel(filepath)

        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)
        assert isinstance(result, dict)
        assert len(result) == 3

        for sheet_name, df in result.items():
            assert sheet_name in sample_multi_sheet_data
            assert isinstance(df, pd.DataFrame)
            assert result[sheet_name].equals(sample_multi_sheet_data[sheet_name])

    @patch('pandas.read_excel')
    def test_read_excel_multi_sheet_specific_sheet_name_string_filepath(
        self, mock_pd_read_excel, sample_multi_sheet_data
    ):
        """Test reading Excel with multiple sheets, specific sheet_name, string filepath."""
        # Setup
        filepath = "/path/to/file.xlsx"
        sheet_name = "Sheet2"
        mock_pd_read_excel.return_value = sample_multi_sheet_data

        result = pandasai.read_excel(filepath, sheet_name=sheet_name)
        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)

        assert isinstance(result, pd.DataFrame)
        assert result.equals(sample_multi_sheet_data[sheet_name])

    @patch('pandas.read_excel')
    def test_read_excel_multi_sheet_specific_sheet_name_bytesio_filepath(
        self, mock_pd_read_excel, sample_multi_sheet_data
    ):
        """Test reading Excel with multiple sheets, specific sheet_name, BytesIO filepath."""
        # Setup
        filepath = BytesIO(b"fake excel content")
        sheet_name = "Sheet1"
        mock_pd_read_excel.return_value = sample_multi_sheet_data

        result = pandasai.read_excel(filepath, sheet_name=sheet_name)
        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)

        assert isinstance(result, pd.DataFrame)
        assert result.equals(sample_multi_sheet_data[sheet_name])

    @patch('pandas.read_excel')
    def test_read_excel_multi_sheet_nonexistent_sheet_name(
        self, mock_pd_read_excel, sample_multi_sheet_data
    ):
        """Test reading Excel with multiple sheets, nonexistent sheet_name."""
        # Setup
        filepath = "/path/to/file.xlsx"
        sheet_name = "NonexistentSheet"
        mock_pd_read_excel.return_value = sample_multi_sheet_data

        result = pandasai.read_excel(filepath, sheet_name=sheet_name)

        # Assert - should return all sheets since the specified sheet doesn't exist
        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)
        assert isinstance(result, dict)
        assert len(result) == 3
        for sheet_name, df in result.items():
            assert sheet_name in sample_multi_sheet_data
            assert isinstance(df, pd.DataFrame)
            assert result[sheet_name].equals(sample_multi_sheet_data[sheet_name])

    @patch('pandas.read_excel')
    def test_read_excel_pandas_exception(self, mock_pd_read_excel):
        """Test that pandas exceptions are propagated."""
        # Setup
        filepath = "/path/to/nonexistent.xlsx"
        mock_pd_read_excel.side_effect = FileNotFoundError("File not found")

        # Execute & Assert
        with pytest.raises(FileNotFoundError, match="File not found"):
            pandasai.read_excel(filepath)

    @patch('pandas.read_excel')
    def test_read_excel_empty_sheet_name_string(
        self, mock_pd_read_excel, sample_multi_sheet_data
    ):
        """Test reading Excel with empty string as sheet_name."""
        # Setup
        filepath = "/path/to/file.xlsx"
        sheet_name = ""
        mock_pd_read_excel.return_value = sample_multi_sheet_data

        result = pandasai.read_excel(filepath, sheet_name=sheet_name)

        # Assert - empty string should be treated as falsy, return all sheets
        mock_pd_read_excel.assert_called_once_with(filepath, sheet_name=None)
        assert isinstance(result, dict)
        assert len(result) == 3
        for sheet_name, df in result.items():
            assert sheet_name in sample_multi_sheet_data
            assert isinstance(df, pd.DataFrame)
            assert result[sheet_name].equals(sample_multi_sheet_data[sheet_name])

    def test_read_excel_type_hints(self):
        """Test that the function signature matches expected types."""
        import inspect

        sig = inspect.signature(pandasai.read_excel)

        # Check parameter names and types
        params = sig.parameters
        assert "filepath" in params
        assert "sheet_name" in params

        # Check that sheet_name has default value
        assert params["sheet_name"].default is None
