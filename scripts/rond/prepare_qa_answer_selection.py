import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


CHOICE_PATTERN = re.compile(r"^([A-Z])\.\s*(.*)$")


def normalize_question(text):
    text = text.strip()
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"\[LC\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_choice(text):
    text = text.strip()
    match = CHOICE_PATTERN.match(text)
    if not match:
        return None, text
    return match.group(1), match.group(2).strip()


def question_pair_text(question, choice_label, choice_text):
    return f"Question: {question}\nCandidate answer ({choice_label}): {choice_text}"


def split_questions(question_rows, seed, train_ratio, val_ratio):
    rng = random.Random(seed)
    items = list(question_rows)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        while n_train + n_val >= n:
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def explode_pairs(question_rows):
    pair_rows = []
    pair_id = 0
    for question_row in question_rows:
        for choice in question_row["choices"]:
            pair_rows.append(
                {
                    "id": pair_id,
                    "question_id": question_row["question_id"],
                    "question": question_row["question"],
                    "choice_label": choice["choice_label"],
                    "choice_text": choice["choice_text"],
                    "text": question_pair_text(
                        question_row["question"],
                        choice["choice_label"],
                        choice["choice_text"],
                    ),
                    "label": "correct" if choice["is_correct"] else "incorrect",
                }
            )
            pair_id += 1
    return pair_rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    grouped = {}
    with Path(args.input).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = normalize_question(row.get("Question", ""))
            choice_label, choice_text = normalize_choice(row.get("Answer_choice", ""))
            correct = row.get("Correct_or_not", "").strip() == "1"
            if not question or not choice_label or not choice_text:
                continue
            grouped.setdefault(question, []).append(
                {
                    "choice_label": choice_label,
                    "choice_text": choice_text,
                    "is_correct": correct,
                }
            )

    question_rows = []
    for idx, (question, choices) in enumerate(grouped.items()):
        choices = sorted(choices, key=lambda item: item["choice_label"])
        if sum(1 for choice in choices if choice["is_correct"]) != 1:
            continue
        question_rows.append(
            {
                "question_id": idx,
                "question": question,
                "choices": choices,
                "num_options": len(choices),
            }
        )

    train_q, val_q, test_q = split_questions(
        question_rows=question_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_rows = explode_pairs(train_q)
    val_rows = explode_pairs(val_q)
    test_rows = explode_pairs(test_q)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    metadata = {
        "source_file": str(Path(args.input)),
        "num_questions": len(question_rows),
        "num_pair_examples": len(train_rows) + len(val_rows) + len(test_rows),
        "question_split_sizes": {
            "train": len(train_q),
            "val": len(val_q),
            "test": len(test_q),
        },
        "pair_split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "pair_label_counts": dict(
            Counter(row["label"] for row in train_rows + val_rows + test_rows)
        ),
        "option_count_distribution": dict(
            Counter(question_row["num_options"] for question_row in question_rows)
        ),
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
