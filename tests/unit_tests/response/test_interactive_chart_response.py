import base64
import io
import json

import pytest

from pandasai.core.response.interactive_chart import InteractiveChartResponse


@pytest.fixture
def sample_json():
    # Create a small test plotly dictionary
    return {
        "data": [
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
                "type": "scatter",
                "mode": "lines+markers",
                "marker": {"color": "red"},
            }
        ],
        "layout": {
            "title": "Test Chart",
            "xaxis": {"title": "X Axis"},
            "yaxis": {"title": "Y Axis"},
        },
        "config": {
            "responsive": True,
            "displayModeBar": True,
            "showSendToCloud": False,
        },
        "image": {
            "width": 100,
            "height": 100,
            "format": "png",
            "data": base64.b64encode(io.BytesIO(b"test_image_data").getvalue()).decode("utf-8"),
        },
    }


@pytest.fixture
def interactive_chart_response(sample_json):
    return InteractiveChartResponse(sample_json, "test_code")


def test_interactive_chart_response_initialization(interactive_chart_response):
    assert interactive_chart_response.type == "ichart"
    assert interactive_chart_response.last_code_executed == "test_code"


def test_get_interactive_chart_from_json(interactive_chart_response):
    chart = interactive_chart_response._get_chart()
    assert isinstance(chart, dict)
    assert chart["image"]["width"] == 100


def test_get_interactive_chart_from_string(sample_json):
    response = InteractiveChartResponse(json.dumps(sample_json), "test_code")
    chart = response._get_chart()
    assert isinstance(chart, dict)
    assert chart["image"]["width"] == 100


def test_get_interactive_chart_from_unsupported_format():
    with pytest.raises(ValueError):
        response = InteractiveChartResponse(1, "test_code")
        response._get_chart()


def test_save_interactive_chart(interactive_chart_response, tmp_path):
    output_path = tmp_path / "output.json"
    interactive_chart_response.save(str(output_path))
    assert output_path.exists()


def test_get_dict_interactive_chart(interactive_chart_response):
    chart_dict = interactive_chart_response.get_dict_image()
    assert isinstance(chart_dict, dict)
    assert "data" in chart_dict
    assert "layout" in chart_dict
    assert "config" in chart_dict
    assert "image" in chart_dict
    assert isinstance(chart_dict["image"], dict)
    assert "data" in chart_dict["image"]
