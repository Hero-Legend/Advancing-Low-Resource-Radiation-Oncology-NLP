import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_text(text):
    text = text.strip()
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"\[LC\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(label):
    return label.strip().lower()


def stratified_split(rows, label_key, seed, train_ratio, val_ratio):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[label_key]].append(row)

    train, val, test = [], [], []
    for items in buckets.values():
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
        else:
            n_train = max(1, n - 1)
            n_val = 0
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


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

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            text = normalize_text(row.get("Question", ""))
            label = normalize_label(row.get("Answer", ""))
            if not text or label not in {"yes", "no"}:
                continue
            rows.append({"id": idx, "text": text, "label": label})

    train, val, test = stratified_split(
        rows=rows,
        label_key="label",
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)

    label_counts = Counter(row["label"] for row in rows)
    metadata = {
        "source_file": str(input_path),
        "num_examples": len(rows),
        "labels": sorted(label_counts),
        "label_counts": dict(label_counts),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
