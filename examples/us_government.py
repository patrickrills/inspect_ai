from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.scorer._classification import _normalize
from inspect_ai.solver import TaskState, generate, prompt_template, use_tools
from inspect_ai.tool import web_search

GOVT_PROMPT_TEMPLATE = """
Provide a concise and accurate answer to the following question about an office holder in the United States government. The answer should only include the last name of the office holder. If more than one person holds the office, provide a comma-separated list of last names in the order they were elected or appointed.

{prompt}

Remember to put your answer on its own line at the end with each last name separated by a comma.
""".strip()


@task
def us_government_knowledge():
    solver = [
        # use_tools(web_search("google")),
        use_tools(web_search("openai")),
        prompt_template(GOVT_PROMPT_TEMPLATE),
        generate(),
    ]

    return Task(
        dataset=[
            Sample(
                input="Who is the current president of the United States?",
                target="Trump",
            ),
            Sample(
                input="Who are the current U.S. Senators from Virginia?",
                target=["Warner", "Kaine"],
            ),
            Sample(
                input="Who are the current U.S. House Representatives from Massachusetts?",
                target=[
                    "Neal",
                    "McGovern",
                    "Lynch",
                    "Keating",
                    "Clark",
                    "Moulton",
                    "Pressley",
                    "Trahan",
                    "Auchincloss",
                ],
            ),
            Sample(
                input="Who are the current Justices of the U.S. Supreme Court?",
                target=[
                    "Thomas",
                    "Roberts",
                    "Alito",
                    "Sotomayor",
                    "Kagan",
                    "Gorsuch",
                    "Kavanaugh",
                    "Barrett",
                    "Jackson",
                ],
            ),
            Sample(
                input="Who is the current Secretary of State of the United States?",
                target="Rubio",
            ),
            Sample(
                input="Who is the top federal prosecutor in Virginia's eastern district Trump appointed in September 2025 and who did they replace?",
                target=[
                    "Siebert",
                    "Halligan",
                ],
                metadata={"ignore_order": True},
            ),
        ],
        solver=solver,
        scorer=ordered_list(),
    )


@scorer(metrics=[accuracy(), stderr()])
def ordered_list():
    async def score(state: TaskState, target: Target):
        answers = [_normalize(a) for a in state.output.completion.split(",")]
        targets = [_normalize(t) for t in target.target]

        if len(answers) != len(targets):
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Expected {len(targets)} items but got {len(answers)} items.",
            )

        if state.metadata.get("ignore_order", False):
            for t in targets:
                if t not in answers:
                    return Score(
                        value=INCORRECT,
                        answer=state.output.completion,
                        explanation=f"Expected item '{t}' not found in answers.",
                    )
        else:
            for i in range(len(targets)):
                if answers[i] != targets[i]:
                    return Score(
                        value=INCORRECT,
                        answer=state.output.completion,
                        explanation=f"Item {i + 1} expected to be '{targets[i]}' but got '{answers[i]}'.",
                    )

        return Score(value=CORRECT, answer=state.output.completion)

    return score
