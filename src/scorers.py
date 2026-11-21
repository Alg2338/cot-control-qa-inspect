from inspect_ai.scorer import Scorer, choice


def cot_control_scorers(
    mode: str = "word_suppression",
    non_reasoning: bool = False,
    output_no_letter_no_grading: bool = False,
    check_meta_discussion: bool = True,
) -> list[Scorer]:
    """Score answer accuracy; compliance and meta-discussion are not implemented yet."""
    return [choice()]
