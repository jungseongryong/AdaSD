from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


LETTERS = ["A", "B", "C", "D"]
OPTION_COLUMNS = ["opa", "opb", "opc", "opd"]


def _clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _answer_index(value) -> int:
    if hasattr(value, "item"):
        value = value.item()
    idx = int(value)
    if 0 <= idx <= 3:
        return idx
    if 1 <= idx <= 4:
        return idx - 1
    raise ValueError(f"Unexpected MedMCQA cop value: {value!r}")


def convert_example(example: dict, answer_format: str = "letter_text") -> dict:
    options = [_clean(example.get(column)) for column in OPTION_COLUMNS]
    answer_idx = _answer_index(example["cop"])
    answer_letter = LETTERS[answer_idx]
    answer_text = options[answer_idx]
    explanation = _clean(example.get("exp"))

    question_lines = [
        f"Question: {_clean(example.get('question'))}",
        "Choices:",
        f"A. {options[0]}",
        f"B. {options[1]}",
        f"C. {options[2]}",
        f"D. {options[3]}",
        "Answer:",
    ]

    response_parts = []
    if explanation:
        response_parts.extend(["Explanation:", explanation, ""])
    if answer_format == "letter":
        response_parts.append(f"Answer: {answer_letter}")
    elif answer_format == "letter_text":
        response_parts.extend(["Final answer:", f"{answer_letter}. {answer_text}"])
    else:
        raise ValueError(f"Unknown answer_format={answer_format!r}")

    return {
        "Question": "\n".join(question_lines),
        "Complex_CoT": explanation,
        "Response": "\n".join(response_parts),
        "answer_letter": answer_letter,
        "answer_text": answer_text,
        "source_id": _clean(example.get("id")),
        "subject_name": _clean(example.get("subject_name")),
        "topic_name": _clean(example.get("topic_name")),
        "choice_type": _clean(example.get("choice_type")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert openlifescienceai/medmcqa to local SFT columns.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-exp", action="store_true")
    parser.add_argument("--min-exp-chars", type=int, default=None)
    parser.add_argument("--max-exp-chars", type=int, default=None)
    parser.add_argument("--answer-format", choices=["letter", "letter_text"], default="letter_text")
    args = parser.parse_args()

    dataset = load_dataset("openlifescienceai/medmcqa", split=args.split)
    if args.require_exp:
        dataset = dataset.filter(lambda example: bool(_clean(example.get("exp"))))
    if args.min_exp_chars is not None:
        dataset = dataset.filter(lambda example: len(_clean(example.get("exp"))) >= args.min_exp_chars)
    if args.max_exp_chars is not None:
        dataset = dataset.filter(lambda example: len(_clean(example.get("exp"))) <= args.max_exp_chars)
    if args.fraction is not None:
        if not 0 < args.fraction <= 1:
            raise ValueError("--fraction must be in (0, 1].")
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(max(1, int(len(dataset) * args.fraction))))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    keep_columns = [
        "Question",
        "Complex_CoT",
        "Response",
        "answer_letter",
        "answer_text",
        "source_id",
        "subject_name",
        "topic_name",
        "choice_type",
    ]
    converted = dataset.map(
        lambda example: convert_example(example, answer_format=args.answer_format),
        remove_columns=dataset.column_names,
    )
    converted = converted.select_columns(keep_columns)

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    converted.save_to_disk(str(output_dir))
    print(f"Saved {len(converted)} examples to {output_dir}")
    print(converted[0])


if __name__ == "__main__":
    main()
