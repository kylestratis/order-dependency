"""
Represents a full dataset of questions with convenience functions.
"""

# Std lib imports
from copy import deepcopy

# Third party imports
import duckdb

# Local imports

MMLU_TEST = "hf://datasets/cais/mmlu/all/test-00000-of-00001.parquet"


class MultipleChoiceQuestion:
    def __init__(self, question: tuple[str, str, list[str], int]):
        self.question_text: str = question[0]
        self.topic: str = question[1]
        self.answers: list[str] = deepcopy(question[2])
        self.original_answers: list[str] = deepcopy(question[2])
        self.correct_answer_index: int = question[3]
        self.original_correct_answer_index: int = question[3]

    @property
    def correct_answer(self) -> str:
        return self.answers[self.correct_answer_index]

    @property
    def gpt_prompt(self) -> list[dict[str, str]]:
        """
        The GPT prompt for the question, from Zheng, et al. (2024)
        """
        messages = []
        messages.append(
            {
                "role": "system",
                "content": f"The following are multiple choice questions about {self.topic}. You should directly answer the question by choosing the correct option.",
            }
        )
        question = f"""
        Question: {self.question_text}
        Options:
        A. {self.answers[0]},
        B. {self.answers[1]}
        C. {self.answers[2]}
        D. {self.answers[3]}
        """
        messages.append({"role": "user", "content": question})
        return messages

    def change_answer(self, new_index: int) -> None:
        if not 0 <= new_index < len(self.answers):
            raise ValueError("New index out of bounds")
        self.answers[self.original_correct_answer_index], self.answers[new_index] = (
            self.answers[new_index],
            self.answers[self.original_correct_answer_index],
        )
        self.correct_answer_index = new_index

    def reset_answers(self) -> None:
        self.answers = deepcopy(self.original_answers)
        self.correct_answer_index = self.original_correct_answer_index


class Dataset:
    def __init__(self, hf_url: str = MMLU_TEST):
        self.hf_url = hf_url
        self.data = None

    def load_data(self, limit: int = 20, random: bool = False) -> None:
        """
        Loads the dataset from the specified HuggingFace URL, optionally limiting the number of rows and randomizing the order.

        Args:
            limit (int, optional): The maximum number of rows to load. Defaults to 20.
            random (bool, optional): Whether to randomize the order of the loaded rows. Defaults to False.
        """
        # TODO validation of URL filetype
        try:
            loaded_file = duckdb.read_parquet(self.hf_url)
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet file: {e}")

        query = "SELECT * FROM loaded_file"
        query_params = []
        if random:
            query += " ORDER BY RANDOM()"
        if limit:
            query += " LIMIT ?"
            query_params.append(limit)
        raw_data = duckdb.execute(query, query_params).fetchall()
        self.data = [MultipleChoiceQuestion(question) for question in raw_data]

    def change_answers(self, new_index: int) -> None:
        for question in self.data:
            question.change_answer(new_index)

    def reset_answers(self) -> None:
        for question in self.data:
            question.reset_answers()
