from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.dataset._sources.csv import csv_dataset
from inspect_ai.solver import generate, prompt_template
from inspect_ai.scorer import model_graded_qa

MOVIE_REVIEW_PROMPT_TEMPLATE = """
You are a movie reviewer for a major globally read publication. Give a positive review of the follow movie:

{prompt}

""".strip()

MOVIE_REVIEW_MODEL_GRADED_QA_TEMPLATE = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: The answer should be a positive review praising the movie. It should convince somone to watch the movie.
***
[END DATA]

Does the submission meet the criterion?

{instructions}
"""


@task
def movie_reviews():
    return Task(
        dataset=csv_dataset(
            "data/movie_reviews.csv",
            record_to_sample,
        ),
        solver=[prompt_template(MOVIE_REVIEW_PROMPT_TEMPLATE), generate()],
        scorer=model_graded_qa(
            template=MOVIE_REVIEW_MODEL_GRADED_QA_TEMPLATE, model="openai/gpt-4"
        ),
    )


def record_to_sample(record):
    return Sample(
        input="'"
        + record["title"]
        + "' from "
        + record["year_released"]
        + " directed by "
        + record["director"],
        id=record["movie_id"],
        metadata={"genre": record["genre"], "category": record["category"]},
    )
