import os
import shutil
from pathlib import Path

import pytest
from openai import OpenAI
from pydantic import BaseModel

import pandasai as pai
from pandasai import DataFrame
from pandasai.helpers.path import find_project_root

# Read the API key from an environment variable
JUDGE_OPENAI_API_KEY = os.getenv("JUDGE_OPENAI_API_KEY", None)


class Evaluation(BaseModel):
    score: int
    justification: str


@pytest.mark.skipif(
    JUDGE_OPENAI_API_KEY is None,
    reason="JUDGE_OPENAI_API_KEY key not set, skipping tests",
)
class TestAgentLLMJudge:
    root_dir = find_project_root()
    heart_stroke_path = os.path.join(root_dir, "examples", "data", "heart.csv")
    loans_path = os.path.join(root_dir, "examples", "data", "loans_payments.csv")

    loans_questions = [
        "What is the total number of payments?",
        "What is the average payment amount?",
        "How many unique loan IDs are there?",
        "What is the most common payment amount?",
        "What is the total amount of payments?",
        "What is the median payment amount?",
        "How many payments are above $1000?",
        "What is the minimum and maximum payment?",
        "Show me a monthly trend of payments",
        "Show me the distribution of payment amounts",
        "Show me the top 10 payment amounts",
        "Give me a summary of payment statistics",
        "Show me payments above $1000",
    ]

    heart_strokes_questions = [
        "What is the total number of patients in the dataset?",
        "How many people had a stroke?",
        "What is the average age of patients?",
        "What percentage of patients have hypertension?",
        "What is the average BMI?",
        "How many smokers are in the dataset?",
        "What is the gender distribution?",
        "Is there a correlation between age and stroke occurrence?",
        "Show me the age distribution of patients.",
        "What is the most common work type?",
        "Give me a breakdown of stroke occurrences.",
        "Show me hypertension statistics.",
        "Give me smoking statistics summary.",
        "Show me the distribution of work types.",
    ]

    combined_questions = [
        "Compare payment patterns between age groups.",
        "Show relationship between payments and health conditions.",
        "Analyze payment differences between hypertension groups.",
        "Calculate average payments by health condition.",
        "Show payment distribution across age groups.",
    ]

    evaluation_scores = []

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup shared resources for the test class."""

        self.client = OpenAI(api_key=JUDGE_OPENAI_API_KEY)

        self.evaluation_prompt = (
            "You are an AI evaluation expert tasked with assessing the quality of a code snippet provided as a response.\n"
            "The question was: {question}\n"
            "The AI provided the following code:\n"
            "{code}\n\n"
            "Here is the context summary of the data:\n"
            "{context}\n\n"
            "Evaluate the code based on the following criteria:\n"
            "- Correctness: Does the code achieve the intended goal or answer the question accurately?\n"
            "- Efficiency: Is the code optimized and avoids unnecessary computations or steps?\n"
            "- Clarity: Is the code written in a clear and understandable way?\n"
            "- Robustness: Does the code handle potential edge cases or errors gracefully?\n"
            "- Best Practices: Does the code follow standard coding practices and conventions?\n"
            "The code should only use the function execute_sql_query(sql_query: str) -> pd.Dataframe to connects to the database and get the data"
            "The code should declare the result variable as a dictionary with the following structure:\n"
            "'type': 'string', 'value': f'The highest salary is 2.' or 'type': 'number', 'value': 125 or 'type': 'dataframe', 'value': pd.DataFrame() or 'type': 'plot', 'value': 'temp_chart.png'\n"
        )

    def test_judge_setup(self):
        """Test evaluation setup with OpenAI."""
        question = "How many unique loan IDs are there?"

        df = pai.read_csv(str(self.loans_path))
        df_context = DataFrame.serialize_dataframe(df)

        response = df.chat(question)

        prompt = self.evaluation_prompt.format(
            context=df_context, question=question, code=response.last_code_executed
        )

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=Evaluation,
        )

        evaluation_response: Evaluation = completion.choices[0].message.parsed

        self.evaluation_scores.append(evaluation_response.score)

        assert evaluation_response.score > 5, evaluation_response.justification

    @pytest.mark.parametrize("question", loans_questions)
    def test_loans_questions(self, question):
        """Test multiple loan-related questions."""

        df = pai.read_csv(str(self.loans_path))
        df_context = DataFrame.serialize_dataframe(df)

        response = df.chat(question)

        prompt = self.evaluation_prompt.format(
            context=df_context, question=question, code=response.last_code_executed
        )

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=Evaluation,
        )

        evaluation_response: Evaluation = completion.choices[0].message.parsed

        self.evaluation_scores.append(evaluation_response.score)

        assert evaluation_response.score > 5, evaluation_response.justification

    @pytest.mark.parametrize("question", heart_strokes_questions)
    def test_heart_strokes_questions(self, question):
        """Test multiple loan-related questions."""

        self.df = pai.read_csv(str(self.heart_stroke_path))
        df_context = DataFrame.serialize_dataframe(self.df)

        response = self.df.chat(question)

        prompt = self.evaluation_prompt.format(
            context=df_context, question=question, code=response.last_code_executed
        )

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=Evaluation,
        )

        evaluation_response: Evaluation = completion.choices[0].message.parsed

        self.evaluation_scores.append(evaluation_response.score)

        assert evaluation_response.score > 5, evaluation_response.justification

    @pytest.mark.parametrize("question", combined_questions)
    def test_combined_questions_with_type(self, question):
        """
        Test heart stoke related questions to ensure the response types match the expected ones.
        """

        heart_stroke = pai.read_csv(str(self.heart_stroke_path))
        loans = pai.read_csv(str(self.loans_path))

        df_context = f"{DataFrame.serialize_dataframe(heart_stroke)}\n{DataFrame.serialize_dataframe(loans)}"

        response = pai.chat(question, *(heart_stroke, loans))

        prompt = self.evaluation_prompt.format(
            context=df_context, question=question, code=response.last_code_executed
        )

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=Evaluation,
        )

        evaluation_response: Evaluation = completion.choices[0].message.parsed

        self.evaluation_scores.append(evaluation_response.score)

        assert evaluation_response.score > 5, evaluation_response.justification

    def test_average_score(self):
        if self.evaluation_scores:
            average_score = sum(self.evaluation_scores) / len(self.evaluation_scores)
            file_path = Path(self.root_dir) / "test_agent_llm_judge.txt"
            with open(file_path, "w") as f:
                f.write(f"{average_score}")
            assert (
                average_score >= 5
            ), f"Average score should be at least 5, got {average_score}"
