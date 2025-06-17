from io import BytesIO

import pandas as pd
import pytest

import pandasai


class TestReadExcel:
    """Test suite for the read_excel function."""

    def test_read_excel_single_sheet_string_filepath(self):
        """Test reading Excel with single sheet and string filepath."""
        # Setup
        filepath = "tests/examples/data/sample_single_sheet_data.xlsx"

        result = pandasai.read_excel(filepath)

        assert isinstance(result, pandasai.DataFrame)

    def test_read_excel_single_sheet_bytesio_filepath(self):
        """Test reading Excel with single sheet and BytesIO filepath."""
        # Setup
        with open("tests/examples/data/sample_single_sheet_data.xlsx", "rb") as f:
            file_content = BytesIO(f.read())

        result = pandasai.read_excel(file_content)

        assert isinstance(result, pandasai.DataFrame)

    def test_read_excel_multi_sheet_unspecified_sheet_name_string_filepath(self):
        """Test reading Excel with multiple sheet and string filepath, without the sheet_name parameter."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        df = pd.read_excel(filepath)

        result = pandasai.read_excel(filepath)

        assert isinstance(result, pandasai.DataFrame)
        assert result.equals(df)

    def test_read_excel_multi_sheet_unspecified_sheet_name_bytesio_filepath(self):
        """Test reading Excel with multiple sheet and BytesIO filepath, without the sheet_name parameter."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        df = pd.read_excel(filepath)

        with open(filepath, "rb") as f:
            file_content = BytesIO(f.read())

        result = pandasai.read_excel(file_content)

        assert isinstance(result, pandasai.DataFrame)
        assert result.equals(df)

    def test_read_excel_multi_sheet_no_sheet_name_string_filepath(self):
        """Test reading Excel with multiple sheets, no sheet_name specified, string filepath."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        df = pd.read_excel(filepath, sheet_name=None)

        result = pandasai.read_excel(filepath, sheet_name=None)

        assert isinstance(result, dict)
        assert len(result) == len(df)

        for sheet_name in result.keys():
            assert sheet_name in df.keys()
            assert isinstance(result[sheet_name], pandasai.DataFrame)
            assert result[sheet_name].equals(df[sheet_name])

    def test_read_excel_multi_sheet_no_sheet_name_bytesio_filepath(self):
        """Test reading Excel with multiple sheets, no sheet_name specified, BytesIO filepath."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        df = pd.read_excel(filepath, sheet_name=None)
        with open(filepath, "rb") as f:
            file_content = BytesIO(f.read())

        # Execute
        result = pandasai.read_excel(file_content, sheet_name=None)

        assert isinstance(result, dict)
        assert len(result) == len(df)

        for sheet_name in result.keys():
            assert sheet_name in df.keys()
            assert isinstance(result[sheet_name], pandasai.DataFrame)
            assert result[sheet_name].equals(df[sheet_name])

    def test_read_excel_multi_sheet_specific_sheet_name_string_filepath(self):
        """Test reading Excel with multiple sheets, specific sheet_name, string filepath."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        sheet_name = "Sheet2"

        result = pandasai.read_excel(filepath, sheet_name=sheet_name)

        assert isinstance(result, pandasai.DataFrame)

    def test_read_excel_multi_sheet_specific_sheet_name_bytesio_filepath(self):
        """Test reading Excel with multiple sheets, specific sheet_name, BytesIO filepath."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        with open(filepath, "rb") as f:
            file_content = BytesIO(f.read())

        sheet_name = "Sheet1"
        result = pandasai.read_excel(file_content, sheet_name=sheet_name)

        assert isinstance(result, pandasai.DataFrame)

    def test_read_excel_multi_sheet_specific_sheet_name_with_space_string_filepath(
        self,
    ):
        """Test reading Excel with multiple sheets, specific sheet_name with space, string filepath."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        sheet_name = "Sheet 2"

        result = pandasai.read_excel(filepath, sheet_name=sheet_name)

        assert isinstance(result, pandasai.DataFrame)

    def test_read_excel_multi_sheet_specific_sheet_name_with_space_bytesio_filepath(
        self,
    ):
        """Test reading Excel with multiple sheets, specific sheet_name with space, BytesIO filepath."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        with open(filepath, "rb") as f:
            file_content = BytesIO(f.read())
        sheet_name = "Sheet 1"

        result = pandasai.read_excel(file_content, sheet_name=sheet_name)

        assert isinstance(result, pandasai.DataFrame)

    def test_read_excel_multi_sheet_nonexistent_sheet_name(self):
        """Test reading Excel with multiple sheets, nonexistent sheet_name."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        sheet_name = "NonexistentSheet"

        with pytest.raises(ValueError):
            pandasai.read_excel(filepath, sheet_name=sheet_name)

    def test_read_excel_pandas_exception(self):
        """Test that pandas exceptions are propagated."""
        # Setup
        filepath = "/path/to/nonexistent.xlsx"

        # Execute & Assert
        with pytest.raises(FileNotFoundError):
            pandasai.read_excel(filepath)

    def test_read_excel_empty_sheet_name_string(self):
        """Test reading Excel with empty string as sheet_name."""
        # Setup
        filepath = "tests/examples/data/sample_multi_sheet_data.xlsx"
        sheet_name = ""

        with pytest.raises(ValueError):
            pandasai.read_excel(filepath, sheet_name=sheet_name)

    def test_read_excel_type_hints(self):
        """Test that the function signature matches expected types."""
        import inspect

        sig = inspect.signature(pandasai.read_excel)

        # Check parameter names and types
        params = sig.parameters
        assert "filepath" in params
        assert "sheet_name" in params

        # Check that sheet_name has default value
        assert params["sheet_name"].default == 0
