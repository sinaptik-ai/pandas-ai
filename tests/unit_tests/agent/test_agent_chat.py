import os
import shutil
from pathlib import Path
from types import UnionType
from typing import List, Tuple

import pytest

import pandasai as pai
from pandasai import DataFrame
from pandasai.core.response import (
    ChartResponse,
    DataFrameResponse,
    NumberResponse,
    StringResponse,
)
from pandasai.helpers.filemanager import find_project_root

# Read the API key from an environment variable
API_KEY = os.getenv("PANDABI_API_KEY_TEST_CHAT", None)


@pytest.mark.skipif(
    API_KEY is None, reason="API key not set, skipping integration tests"
)
class TestAgentChat:
    root_dir = find_project_root()
    heart_stroke_path = os.path.join(root_dir, "examples", "data", "heart.csv")
    loans_path = os.path.join(root_dir, "examples", "data", "loans_payments.csv")
    numeric_questions_with_answer = [
        ("What is the total quantity sold across all products and regions?", 105),
        ("What is the correlation coefficient between Sales and Profit?", 1.0),
        (
            "What is the standard deviation of daily sales for the entire dataset?",
            231.0,
        ),
        (
            "Give me the number of the highest average profit margin among all regions?",
            0.2,
        ),
        (
            "What is the difference in total Sales between Product A and Product B across the entire dataset?",
            700,
        ),
        ("Over the entire dataset, how many days had sales above 900?", 5),
        (
            "What was the year-over-year growth in total sales from 2022 to 2023 (in percent)?",
            7.84,
        ),
    ]
    loans_questions_with_type: List[Tuple[str, type | UnionType]] = [
        ("What is the total number of payments?", NumberResponse),
        ("What is the average payment amount?", NumberResponse),
        ("How many unique loan IDs are there?", NumberResponse),
        ("What is the most common payment amount?", NumberResponse),
        ("What is the total amount of payments?", NumberResponse),
        ("What is the median payment amount?", NumberResponse),
        ("How many payments are above $1000?", NumberResponse),
        (
            "What is the minimum and maximum payment?",
            (NumberResponse, DataFrameResponse),
        ),
        ("Show me a monthly trend of payments", (ChartResponse, DataFrameResponse)),
        (
            "Show me the distribution of payment amounts",
            (ChartResponse, DataFrameResponse),
        ),
        ("Show me the top 10 payment amounts", DataFrameResponse),
        (
            "Give me a summary of payment statistics",
            (StringResponse, DataFrameResponse),
        ),
        ("Show me payments above $1000", DataFrameResponse),
    ]
    heart_strokes_questions_with_type: List[Tuple[str, type | UnionType]] = [
        ("What is the total number of patients in the dataset?", NumberResponse),
        ("How many people had a stroke?", NumberResponse),
        ("What is the average age of patients?", NumberResponse),
        ("What percentage of patients have hypertension?", NumberResponse),
        ("What is the average BMI?", NumberResponse),
        ("How many smokers are in the dataset?", NumberResponse),
        ("What is the gender distribution?", (ChartResponse, DataFrameResponse)),
        (
            "Is there a correlation between age and stroke occurrence?",
            (ChartResponse, StringResponse),
        ),
        (
            "Show me the age distribution of patients",
            (ChartResponse, DataFrameResponse),
        ),
        ("What is the most common work type?", StringResponse),
        (
            "Give me a breakdown of stroke occurrences",
            (StringResponse, DataFrameResponse),
        ),
        ("Show me hypertension statistics", (StringResponse, DataFrameResponse)),
        ("Give me smoking statistics summary", (StringResponse, DataFrameResponse)),
        ("Show me the distribution of work types", (ChartResponse, DataFrameResponse)),
    ]
    combined_questions_with_type: List[Tuple[str, type | UnionType]] = [
        (
            "Compare payment patterns between age groups",
            (ChartResponse, DataFrameResponse),
        ),
        (
            "Show relationship between payments and health conditions",
            (ChartResponse, DataFrameResponse),
        ),
        (
            "Analyze payment differences between hypertension groups",
            (StringResponse, DataFrameResponse),
        ),
        (
            "Calculate average payments by health condition",
            (NumberResponse, DataFrameResponse),
        ),
        (
            "Show payment distribution across age groups",
            (ChartResponse, DataFrameResponse),
        ),
    ]

    @pytest.fixture
    def pandas_ai(self):
        pai.api_key.set(API_KEY)
        return pai

    @pytest.mark.parametrize("question,expected", numeric_questions_with_answer)
    def test_numeric_questions(self, question, expected, pandas_ai):
        """
        Test numeric questions to ensure the response match the expected ones.
        """

        # Sample DataFrame spanning two years (2022-2023), multiple regions and products
        df = DataFrame(
            {
                "Date": [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-02-01",
                    "2022-02-02",
                    "2022-02-03",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-02-01",
                    "2023-02-02",
                    "2023-02-03",
                ],
                "Region": [
                    "North",
                    "North",
                    "South",
                    "South",
                    "East",
                    "East",
                    "North",
                    "North",
                    "South",
                    "South",
                    "East",
                    "East",
                ],
                "Product": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                "Sales": [
                    1000,
                    800,
                    1200,
                    900,
                    500,
                    700,
                    1100,
                    850,
                    1250,
                    950,
                    600,
                    750,
                ],
                "Profit": [200, 160, 240, 180, 100, 140, 220, 170, 250, 190, 120, 150],
                "Quantity": [10, 8, 12, 9, 5, 7, 11, 8, 13, 9, 6, 7],
            }
        )

        response = pandas_ai.chat(question, df)

        assert isinstance(
            response, NumberResponse
        ), f"Expected a NumberResponse, got {type(response)} for question: {question}"

        model_value = float(response.value)

        assert model_value == pytest.approx(expected, abs=0.5), (
            f"Question: {question}\n" f"Expected: {expected}, Got: {model_value}"
        )

    @pytest.mark.parametrize("question,expected", loans_questions_with_type)
    def test_loans_questions_type(self, question, expected, pandas_ai):
        """
        Test loan-related questions to ensure the response types match the expected ones.
        """

        df = pandas_ai.read_csv(str(self.loans_path))

        response = pandas_ai.chat(question, df)

        assert isinstance(
            response, expected
        ), f"Expected type {expected}, got {type(response)} for question: {question}"

    @pytest.mark.parametrize("question,expected", heart_strokes_questions_with_type)
    def test_heart_strokes_questions_type(self, question, expected, pandas_ai):
        """
        Test heart stoke related questions to ensure the response types match the expected ones.
        """

        df = pandas_ai.read_csv(str(self.heart_stroke_path))

        response = pandas_ai.chat(question, df)

        assert isinstance(
            response, expected
        ), f"Expected type {expected}, got {type(response)} for question: {question}"

    @pytest.mark.parametrize("question,expected", combined_questions_with_type)
    def test_combined_questions_with_type(self, question, expected, pandas_ai):
        """
        Test heart stoke related questions to ensure the response types match the expected ones.
        """

        heart_stroke = pandas_ai.read_csv(str(self.heart_stroke_path))
        loans = pandas_ai.read_csv(str(self.loans_path))

        response = pandas_ai.chat(question, *(heart_stroke, loans))

        assert isinstance(
            response, expected
        ), f"Expected type {expected}, got {type(response)} for question: {question}"
