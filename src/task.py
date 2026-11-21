from inspect_ai import Task, task

from adversarial_prompts import ADVERSARIAL_PRESSURES
from dataset import DatasetName, load_cot_control_dataset
from prompts import CONTROL_PROMPTS
from solver import cot_control_solver
from scorers import cot_control_scorers


@task
def cot_control_qa(
    data: DatasetName = "gpqa",
    mode: str = "word_suppression",
    non_reasoning: bool = False,
    output_no_letter_no_grading: bool = False,
    adversarial_pressure: str = "na",
    custom_adversarial_prompt: str | None = None,
    monitor_aware: bool = False,
    eval_aware: bool = False,
    check_meta_discussion: bool = True,
) -> Task:
    """Evaluate answer accuracy and compliance with a CoT constraint."""
    valid_modes = ("baseline", *CONTROL_PROMPTS)
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown mode: {mode!r}. Expected one of: {', '.join(valid_modes)}"
        )
    if adversarial_pressure not in ADVERSARIAL_PRESSURES:
        raise ValueError(
            f"Unknown adversarial_pressure: {adversarial_pressure!r}. "
            f"Expected one of: {', '.join(ADVERSARIAL_PRESSURES)}"
        )

    dataset = load_cot_control_dataset(data=data)

    solver = cot_control_solver(
        mode=mode,
        non_reasoning=non_reasoning,
        output_no_letter_no_grading=output_no_letter_no_grading,
        adversarial_pressure=adversarial_pressure,
        custom_adversarial_prompt=custom_adversarial_prompt,
        monitor_aware=monitor_aware,
        eval_aware=eval_aware,
    )

    scorers = cot_control_scorers(
        mode=mode,
        non_reasoning=non_reasoning,
        output_no_letter_no_grading=output_no_letter_no_grading,
        check_meta_discussion=check_meta_discussion,
    )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
    )
