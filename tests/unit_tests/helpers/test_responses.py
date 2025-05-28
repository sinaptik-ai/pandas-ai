import base64
import io
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from PIL import Image

import pytest

from pandasai.core.response import (
    ChartResponse,
    DataFrameResponse,
    InteractiveChartResponse,
    NumberResponse,
    StringResponse,
)
from pandasai.core.response.parser import ResponseParser
from pandasai.exceptions import InvalidOutputValueMismatch


class TestResponseParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_parser = ResponseParser()

    def test_parse_valid_number(self):
        result = {"type": "number", "value": 42}
        response = self.response_parser.parse(result)
        self.assertIsInstance(response, NumberResponse)
        self.assertEqual(response.value, 42)
        self.assertEqual(response.last_code_executed, None)
        self.assertEqual(response.type, "number")

    def test_parse_valid_string(self):
        result = {"type": "string", "value": "test string"}
        response = self.response_parser.parse(result)
        self.assertIsInstance(response, StringResponse)
        self.assertEqual(response.value, "test string")
        self.assertEqual(response.last_code_executed, None)
        self.assertEqual(response.type, "string")

    def test_parse_valid_dataframe(self):
        expected_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = {"type": "dataframe", "value": expected_df}

        response = self.response_parser.parse(result)
        self.assertIsInstance(response, DataFrameResponse)
        pd.testing.assert_frame_equal(response.value, expected_df)
        self.assertEqual(response.last_code_executed, None)
        self.assertEqual(response.type, "dataframe")

    def test_parse_valid_plot(self):
        result = {"type": "plot", "value": "path/to/plot.png"}
        response = self.response_parser.parse(result)
        self.assertIsInstance(response, ChartResponse)
        self.assertEqual(response.value, "path/to/plot.png")
        self.assertEqual(response.last_code_executed, None)
        self.assertEqual(response.type, "chart")

    def test_parse_valid_interactive_plot(self):
        path = "path/to/interactive_plot.json"
        # mock os.path.exists to return True for the plot path
        with patch("os.path.exists", return_value=True) as mock_exists:
            result = {"type": "iplot", "value": path}
            response = self.response_parser.parse(result)
            self.assertIsInstance(response, InteractiveChartResponse)
            self.assertEqual(response.value, path)
            self.assertEqual(response.last_code_executed, None)
            self.assertEqual(response.type, "ichart")

            mock_exists.assert_called_once_with(path)

    def test_parse_invalid_interactive_plot_because_of_not_existing_file(self):
        path = "path/to/interactive_plot.json"
        with pytest.raises(ValueError):
            result = {"type": "iplot", "value": path}
            self.response_parser.parse(result)

    def test_plot_img_show_triggered(self):
        result = {
            "type": "plot",
            "value": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
        }
        response = self.response_parser.parse(result)

        mock_image = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "PIL.Image.open", return_value=mock_image
        ) as mock_open:
            response.show()
            mock_open.assert_called_once()
            mock_image.show.assert_called_once()

        mock_image = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "PIL.Image.open", return_value=mock_image
        ) as mock_open:
            print(response)
            mock_open.assert_called_once()
            mock_image.show.assert_called_once()

    def test_parse_with_last_code_executed(self):
        result = {"type": "number", "value": 42}
        last_code = "print('Hello, World!')"
        response = self.response_parser.parse(result, last_code)
        self.assertIsInstance(response, NumberResponse)
        self.assertEqual(response.value, 42)
        self.assertEqual(response.last_code_executed, last_code)
        self.assertEqual(response.type, "number")

    def test_parse_invalid_type(self):
        result = {"type": "unknown", "value": "test"}
        with self.assertRaises(InvalidOutputValueMismatch):
            self.response_parser.parse(result)

    def test_parse_missing_type(self):
        result = {"value": "test"}
        with self.assertRaises(InvalidOutputValueMismatch):
            self.response_parser.parse(result)

    def test_parse_missing_value(self):
        result = {"type": "string"}
        with self.assertRaises(InvalidOutputValueMismatch):
            self.response_parser.parse(result)

    def test_validate_invalid_number_type(self):
        result = {"type": "number", "value": "not a number"}
        with self.assertRaises(InvalidOutputValueMismatch):
            self.response_parser._validate_response(result)

    def test_validate_invalid_string_type(self):
        result = {"type": "string", "value": 123}
        with self.assertRaises(InvalidOutputValueMismatch):
            self.response_parser._validate_response(result)

    def test_validate_invalid_dataframe_type(self):
        result = {"type": "dataframe", "value": "not a dataframe"}
        with self.assertRaises(InvalidOutputValueMismatch):
            self.response_parser._validate_response(result)

    def test_validate_invalid_plot_type(self):
        result = {"type": "plot", "value": 12345}
        with self.assertRaises(InvalidOutputValueMismatch):
            self.response_parser._validate_response(result)

    def test_validate_plot_with_base64(self):
        result = {"type": "plot", "value": "data:image/png;base64 fake_image_data"}
        self.assertTrue(self.response_parser._validate_response(result))

    def test_validate_valid_plot_path(self):
        result = {"type": "plot", "value": "/valid/path/to/plot.png"}
        self.assertTrue(self.response_parser._validate_response(result))

    @patch("pandasai.core.response.chart.Image.open")  # Mock the Image.open method
    def test_get_base64_image(self, mock_image_open):
        # Create a mock image
        mock_image = MagicMock(spec=Image.Image)
        mock_image.save = MagicMock()  # Mock the save method
        mock_image_open.return_value = mock_image  # Mock return value for Image.open

        # Create a mock image file path
        mock_image_path = "test_image.png"

        # Initialize ChartResponse with a mock image path
        chart_response = ChartResponse(
            value=mock_image_path, last_code_executed="test_code"
        )

        # Mock the image bytes to be encoded
        mock_image_bytes = io.BytesIO()
        mock_image_bytes.write(b"mock_image_data")
        mock_image_bytes.seek(0)

        def save_to_mock_bytes(file_obj, format=None):
            file_obj.write(mock_image_bytes.read())

        mock_image.save.side_effect = save_to_mock_bytes  # Mock save to write bytes

        # Call the method
        result = chart_response.get_base64_image()

        # Prepare the expected base64 string
        expected_base64 = base64.b64encode(b"mock_image_data").decode("utf-8")

        # Assert the result
        assert result == expected_base64
        mock_image_open.assert_called_once_with(
            mock_image_path
        )  # Ensure the image was opened
        mock_image.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
