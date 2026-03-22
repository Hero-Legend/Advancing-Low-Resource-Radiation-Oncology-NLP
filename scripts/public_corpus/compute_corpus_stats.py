import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) if args.tokenizer else None

    docs = 0
    chars = 0
    tokens = 0
    token_lengths = []

    for row in iter_jsonl(args.input_jsonl):
        text = row.get("text", "")
        docs += 1
        chars += len(text)
        if tokenizer is not None:
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            tokens += length
            token_lengths.append(length)

    payload = {
        "input_jsonl": args.input_jsonl,
        "document_count": docs,
        "character_count": chars,
        "token_count": tokens if tokenizer is not None else None,
        "avg_tokens_per_document": (tokens / docs) if tokenizer is not None and docs else None,
        "max_tokens_per_document": max(token_lengths) if token_lengths else None,
        "min_tokens_per_document": min(token_lengths) if token_lengths else None,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
