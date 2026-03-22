import argparse
import json
import random
import re
from pathlib import Path

import requests


DEFAULT_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_eval_text(record: dict) -> str:
    question = normalize_text(record.get("QUESTION", ""))
    contexts = [normalize_text(x) for x in record.get("CONTEXTS", []) if normalize_text(x)]
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if contexts:
        parts.append("Context: " + " ".join(contexts))
    return "\n".join(parts).strip()


def build_replay_text(record: dict) -> str:
    question = normalize_text(record.get("QUESTION", ""))
    contexts = [normalize_text(x) for x in record.get("CONTEXTS", []) if normalize_text(x)]
    long_answer = normalize_text(record.get("LONG_ANSWER", ""))
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if contexts:
        parts.append("Context: " + " ".join(contexts))
    if long_answer:
        parts.append(f"Explanation: {long_answer}")
    return "\n".join(parts).strip()


def stratified_split(rows, train_frac, val_frac, seed):
    rng = random.Random(seed)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    train_rows, val_rows, test_rows = [], [], []
    for label, group in sorted(by_label.items()):
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train:n_train + n_val])
        test_rows.extend(group[n_train + n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--replay-train-jsonl", required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--stats-json", required=True)
    args = parser.parse_args()

    payload = requests.get(args.source_url, timeout=120).json()
    rows = []
    for pmid, record in payload.items():
        label = normalize_text(record.get("final_decision", "")).lower()
        if label not in {"yes", "no", "maybe"}:
            continue
        eval_text = build_eval_text(record)
        replay_text = build_replay_text(record)
        if len(eval_text) < args.min_chars or len(replay_text) < args.min_chars:
            continue
        rows.append(
            {
                "pmid": str(pmid),
                "label": label,
                "eval_text": eval_text,
                "replay_text": replay_text,
            }
        )

    train_rows, val_rows, test_rows = stratified_split(rows, args.train_frac, args.val_frac, args.seed)

    eval_train = [{"id": row["pmid"], "text": row["eval_text"], "label": row["label"]} for row in train_rows]
    eval_val = [{"id": row["pmid"], "text": row["eval_text"], "label": row["label"]} for row in val_rows]
    eval_test = [{"id": row["pmid"], "text": row["eval_text"], "label": row["label"]} for row in test_rows]
    replay_train = [
        {
            "source": "pubmedqa_pqal_train",
            "doc_id": row["pmid"],
            "title": row["pmid"],
            "url": args.source_url,
            "text": row["replay_text"],
            "char_count": len(row["replay_text"]),
        }
        for row in train_rows
    ]

    write_jsonl(args.train_jsonl, eval_train)
    write_jsonl(args.val_jsonl, eval_val)
    write_jsonl(args.test_jsonl, eval_test)
    write_jsonl(args.replay_train_jsonl, replay_train)

    def counts(rows_):
        out = {}
        for row in rows_:
            out[row["label"]] = out.get(row["label"], 0) + 1
        return out

    stats = {
        "source_url": args.source_url,
        "seed": args.seed,
        "total_examples": len(rows),
        "train_examples": len(eval_train),
        "val_examples": len(eval_val),
        "test_examples": len(eval_test),
        "label_distribution_total": counts(rows),
        "label_distribution_train": counts(train_rows),
        "label_distribution_val": counts(val_rows),
        "label_distribution_test": counts(test_rows),
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "test_jsonl": args.test_jsonl,
        "replay_train_jsonl": args.replay_train_jsonl,
    }
    stats_path = Path(args.stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
