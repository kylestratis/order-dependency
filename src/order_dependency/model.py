# Std lib imports

# Third party imports
from openai import OpenAI


class Model:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def ask_question(self, question: "MultipleChoiceQuestion") -> str:  # type: ignore
        match self.model_name:
            case "gpt-3.5-turbo" | "gpt-4o":
                return self._ask_gpt(question)
            case _:
                raise ValueError(f"Unknown model name: {self.model_name}")

    def _ask_gpt(self, question: "MultipleChoiceQuestion") -> str:  # type: ignore
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=question.gpt_prompt,
        )
        return response.choices[0].text
