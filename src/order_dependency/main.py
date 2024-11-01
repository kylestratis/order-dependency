"""
Application code to run analyses
"""

# Std lib imports
from enum import Enum
from typing import TypedDict
import json
import time

# Third party imports
import click
from dotenv import load_dotenv

# Internal imports
from order_dependency.dataset import Dataset
from order_dependency.model import Model

load_dotenv()


class Ordering(Enum):
    BASELINE = "baseline"
    ALL_A = 0
    ALL_B = 1
    ALL_C = 2
    ALL_D = 3


class IndividualResult(TypedDict):
    question: dict  # dictionary representation of MCQ
    ordering: str  # Ordering.name
    llm_response: int  # should be converted from alpha ID to int position
    llm_response_alpha: str
    discrepancy: bool


class AnalysisResult(TypedDict):
    model_name: str
    individual_results: list[IndividualResult]
    ordering: Ordering
    discrepancy_count: int
    accuracy: float
    # r_std: float


def _setup(model_name: str, data_limit: int, random: bool) -> tuple[Dataset, Model]:
    dataset = Dataset()
    dataset.load_data(limit=data_limit, random=random)
    model = Model(model_name=model_name)
    return dataset, model


def _run_single_analysis(
    dataset: Dataset, model: Model, ordering: Ordering
) -> AnalysisResult:
    individual_results = []

    if ordering != Ordering.BASELINE:
        dataset.change_answers(ordering.value)

    for question in dataset.data:
        pred_answer = model.ask_question(question)  # Should be A, B, C, D
        if pred_answer not in ["A", "B", "C", "D"]:
            pred_answer_idx = None
        else:
            pred_answer_idx = ord(pred_answer) - 65
        individual_results.append(
            IndividualResult(
                question=question.to_dict(),
                ordering=ordering.name,
                llm_response=pred_answer_idx,
                llm_response_alpha=pred_answer,
                discrepancy=question.correct_answer_index != pred_answer_idx,
            )
        )
    return AnalysisResult(
        model_name=model.model_name,
        individual_results=individual_results,
        ordering=ordering.name,
        discrepancy_count=sum(result["discrepancy"] for result in individual_results),
        accuracy=_calculate_accuracy(individual_results),
    )


def _calculate_accuracy(results: list[IndividualResult]) -> float:
    return (
        [result["discrepancy"] for result in results].count(False) / len(results) * 100
    )


def _calculate_r_std(results: list[IndividualResult]) -> float:
    raise NotImplementedError


def _export_results(results: list[AnalysisResult]) -> None:
    with open(
        f"output/run_{time.time()}_{len(results[0]['individual_results'])}_Qs.jsonl",
        "w",
    ) as f:
        json.dump(results, f)


@click.command()
@click.option("--model_name", default="gpt-3.5-turbo", help="Model name")
@click.option("--data_limit", default=20, help="Number of questions to use")
@click.option("--random", default=True, help="Whether to use random questions")
def run_full_analysis(
    model_name: str = "gpt-3.5-turbo", data_limit: int = 20, random: bool = True
) -> None:
    dataset, model = _setup(model_name, data_limit, random)
    results: list[AnalysisResult] = [
        _run_single_analysis(dataset, model, ordering) for ordering in Ordering
    ]
    click.echo(f"Results: {results}")
    _export_results(results)
