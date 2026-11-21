from ast import literal_eval
from pathlib import Path
from typing import Literal

from inspect_ai.dataset import Dataset, Sample, csv_dataset


DatasetName = Literal["mmlu-pro_mini", "hle", "gpqa"]

DATASET_FILES = {
    "mmlu-pro_mini": "mmlu_pro_mini_w_keyword.csv",
    "hle": "hle_w_keyword.csv",
    "gpqa": "gpqa_w_keyword.csv",
}


def _record_to_sample(record: dict[str, str]) -> Sample:
    choices = literal_eval(record.get("options") or record["answer_options"])
    answer = record["answer"]
    # HLE stores answer letters; GPQA and MMLU-Pro store answer text.
    if answer in choices:
        target = chr(ord("A") + choices.index(answer))
    elif answer in [chr(ord("A") + i) for i in range(len(choices))]:
        target = answer
    else:
        raise ValueError(f"Answer {answer!r} does not match the choices")

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"question", "options", "answer_options", "answer"}
    }
    for key in ("valid_keywords", "keywords_with_synonyms"):
        metadata[key] = literal_eval(record[key])
    metadata["answer_text"] = choices[ord(target) - ord("A")]

    return Sample(
        input=record["question"],
        choices=choices,
        target=target,
        metadata=metadata,
    )


def load_cot_control_dataset(data: DatasetName = "gpqa") -> Dataset:
    """Load a bundled dataset independently of the experiment mode."""
    if data not in DATASET_FILES:
        raise ValueError(
            f"Unknown dataset: {data!r}. "
            f"Expected one of: {', '.join(DATASET_FILES)}"
        )

    path = Path(__file__).resolve().parent.parent / "datasets" / DATASET_FILES[data]
    return csv_dataset(str(path), sample_fields=_record_to_sample, name=data)
