import json
import os
from typing import Any

from .base import BaseResponse


class InteractiveChartResponse(BaseResponse):
    def __init__(self, value: Any, last_code_executed: str):
        super().__init__(value, "ichart", last_code_executed)

        self._validate()

    def _get_chart(self) -> dict:
        if isinstance(self.value, dict):
            return self.value

        if isinstance(self.value, str):
            if os.path.exists(self.value):
                with open(self.value, "rb") as f:
                    return json.load(f)

            return json.loads(self.value)

        raise ValueError(
            "Invalid value type for InteractiveChartResponse. Expected dict or str."
        )

    def save(self, path: str):
        img = self._get_chart()
        with open(path, "w") as f:
            json.dump(img, f)

    def __str__(self) -> str:
        return self.value if isinstance(self.value, str) else json.dumps(self.value)

    def get_dict_image(self) -> dict:
        return self._get_chart()

    def _validate(self):
        if not isinstance(self.value, (dict, str)):
            raise ValueError(
                "InteractiveChartResponse value must be a dict or a str representing a file path."
            )

        # if a string, it can be a path to a file or a JSON string
        if isinstance(self.value, str):
            try:
                json.loads(self.value)  # Check if it's a valid JSON string
            except json.JSONDecodeError:
                # If it fails, check if it's a valid file path
                if not os.path.exists(self.value):
                    raise ValueError(
                        "InteractiveChartResponse value must be a valid file path or a JSON string."
                    )
