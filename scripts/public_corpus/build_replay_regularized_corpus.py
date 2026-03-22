import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-jsonl", required=True)
    parser.add_argument("--replay-jsonl", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--replay-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-stats-json", required=True)
    parser.add_argument("--output-txt", default="")
    args = parser.parse_args()

    if not 0 < args.replay_ratio < 1:
        raise ValueError("--replay-ratio must be between 0 and 1")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    focused_docs = list(iter_jsonl(args.focused_jsonl))
    replay_docs = list(iter_jsonl(args.replay_jsonl))

    focused_token_total = sum(token_length(tokenizer, row.get("text", "")) for row in focused_docs)
    target_replay_tokens = int(round(focused_token_total * args.replay_ratio / (1 - args.replay_ratio)))

    rng = random.Random(args.seed)
    shuffled_replay = replay_docs[:]
    rng.shuffle(shuffled_replay)

    selected = []
    replay_token_total = 0
    for row in shuffled_replay:
        text = row.get("text", "")
        length = token_length(tokenizer, text)
        selected.append(
            {
                **row,
                "token_count": length,
            }
        )
        replay_token_total += length
        if replay_token_total >= target_replay_tokens:
            break

    combined = []
    for row in focused_docs:
        combined.append({**row, "replay_group": "focused"})
    for row in selected:
        row = dict(row)
        row.pop("token_count", None)
        row["replay_group"] = "replay"
        combined.append(row)

    rng.shuffle(combined)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output_txt:
        output_txt = Path(args.output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text("\n\n".join(row["text"] for row in combined), encoding="utf-8")

    actual_ratio = replay_token_total / max(focused_token_total + replay_token_total, 1)
    stats = {
        "focused_jsonl": args.focused_jsonl,
        "replay_jsonl": args.replay_jsonl,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "requested_replay_ratio": args.replay_ratio,
        "focused_document_count": len(focused_docs),
        "replay_candidate_count": len(replay_docs),
        "selected_replay_documents": len(selected),
        "focused_token_total": focused_token_total,
        "target_replay_tokens": target_replay_tokens,
        "actual_replay_tokens": replay_token_total,
        "actual_replay_ratio": actual_ratio,
        "combined_document_count": len(combined),
        "output_jsonl": str(output_jsonl),
    }
    output_stats = Path(args.output_stats_json)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    output_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
