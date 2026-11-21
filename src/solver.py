from inspect_ai.solver import Generate, Solver, TaskState, solver

from adversarial_prompts import ADVERSARIAL_PRESSURES
from prompts import (
    ANSWER_THEN_EXPLAIN,
    BASELINE_ANSWER_INSTRUCTION,
    BASELINE_REASONING_INSTRUCTION,
    BASELINE_TEMPLATE,
    CONTROL_PROMPTS,
    EVAL_AWARE_INSTRUCTION,
    MONITOR_AWARE_INSTRUCTION,
    OUTPUT_REASONING_INSTRUCTION,
)


@solver
def cot_control_solver(
    mode: str = "word_suppression",
    non_reasoning: bool = False,
    output_no_letter_no_grading: bool = False,
    adversarial_pressure: str = "na",
    custom_adversarial_prompt: str | None = None,
    monitor_aware: bool = False,
    eval_aware: bool = False,
) -> Solver:
    """Assemble the question and control instructions for a single generation."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        is_gpt_oss = "gpt-oss" in str(state.model).lower()
        reasoning_term = "analysis channel" if is_gpt_oss else "reasoning stage"
        channel_term = "analysis channel" if is_gpt_oss else "reasoning"
        instructions = []
        simple = output_no_letter_no_grading and mode != "baseline"
        variant = "simple" if simple else "cot" if non_reasoning else "reasoning"
        if mode != "baseline":
            control_value = (
                state.metadata["control_value"]
                if "{control_value}" in CONTROL_PROMPTS[mode][variant]
                else ""
            )
            instructions.append(
                "Requirement: " + CONTROL_PROMPTS[mode][variant].format(
                    control_value=control_value,
                    reasoning_term=reasoning_term,
                )
            )
            if simple:
                pressure = custom_adversarial_prompt or ""
            elif non_reasoning:
                pressure = ADVERSARIAL_PRESSURES[adversarial_pressure]
            else:
                pressure = (
                    custom_adversarial_prompt
                    if custom_adversarial_prompt is not None
                    else ADVERSARIAL_PRESSURES[adversarial_pressure]
                )
            if simple and mode != "ignore_question":
                instructions.insert(0, ANSWER_THEN_EXPLAIN)
            elif not simple and non_reasoning:
                instructions.append(OUTPUT_REASONING_INSTRUCTION)
            if pressure:
                instructions.append(
                    pressure.format(
                        reasoning_term=channel_term,
                        reasoning_term_upper=channel_term.upper(),
                    )
                )
        if monitor_aware:
            instructions.append(
                MONITOR_AWARE_INSTRUCTION.format(channel_term=channel_term)
            )
        if eval_aware:
            instructions.append(EVAL_AWARE_INSTRUCTION)

        state.user_prompt.text = BASELINE_TEMPLATE.format(
            prompt=state.user_prompt.text,
            choices="\n".join(
                f"{chr(ord('A') + i)}) {option.value}"
                for i, option in enumerate(state.choices)
            ),
        )
        if mode == "baseline" and non_reasoning:
            state.user_prompt.text += BASELINE_REASONING_INSTRUCTION
        if not simple:
            state.user_prompt.text += BASELINE_ANSWER_INSTRUCTION
        if instructions:
            state.user_prompt.text += "\n\n" + "\n\n".join(instructions)
        return await generate(state)

    return solve
