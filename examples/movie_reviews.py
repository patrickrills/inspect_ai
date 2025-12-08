from inspect_ai import Task, task
from inspect_ai.dataset import Sample
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
[Criterion]: The answer should be a positive review praising the movie. It should convince somone to watch the movie. It should also contains a reference to the muppets.
***
[END DATA]

Does the submission meet the criterion?

{instructions}
"""


@task
def movie_reviews():
    return Task(
        dataset=[
            Sample(
                input="'The Matrix' from 1999 directed by the Wachowskis. \n Compare Neo from the Matrix to Fozzie Bear."
            ),
            Sample(input="'The Room' from 2003 directed by Tommy Wiseau"),
        ],
        solver=[prompt_template(MOVIE_REVIEW_PROMPT_TEMPLATE), generate()],
        scorer=model_graded_qa(
            template=MOVIE_REVIEW_MODEL_GRADED_QA_TEMPLATE, model="openai/gpt-4"
        ),
    )
