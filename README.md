# TODOS - Auth to HF (huggingface cli login) and OpenAI (openai.api_key)

This is a Python application that allows you to evaluate how well LLMs handle the order dependency problem, drawing heavily from the paper [Large Language Models Are not Robust Multiple Choice Selectors](https://openreview.net/pdf?id=shr9PXz7T0) by Zheng, et al. It is written to easily handle multiple models with limited extra work.

# Installation
The easiest way to install the project is by using `pip` or `uv` to install the wheel file in `dist/`. You may also clone the repo and use `uv` or any other
`pyproject.toml`-compatible tool to install dependencies to a local virtual environment and then run it.

# Usage
Upon installation, you can run the application either by using `uv run run_analysis` or `run_analysis`. Outpus are saved to the `outputs` directory.

```
Usage: run_analysis [OPTIONS]

Options:
  --model_name TEXT     Model name
  --data_limit INTEGER  Number of questions to use
  --random BOOLEAN      Whether to use random questions
  --help                Show this message and exit.


# Authorization
## Data
To access the Huggingface dataset, use the Huggingface CLI to authenticate your account.

## OpenAI
To access the OpenAI API, set your OpenAI API key in your own `.env` file.

# Future Work
This project could be extended in many ways. It could be extended to handle more models, more MCQ datasets, and more evaluation metrics, such as the standard deviation of the recall balance. It could also be extended to handle more complex questions, such as those that require multiple steps to solve (and evaluating chain-of-thought prompting to answer these).

Testing, which is non-existent in this project, would also be a good addition.