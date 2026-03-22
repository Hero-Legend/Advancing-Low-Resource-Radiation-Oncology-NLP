import argparse
import json
import re
from pathlib import Path

import requests


DEFAULT_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_text(record: dict) -> str:
    question = normalize_text(record.get("QUESTION", ""))
    contexts = [normalize_text(x) for x in record.get("CONTEXTS", []) if normalize_text(x)]
    long_answer = normalize_text(record.get("LONG_ANSWER", ""))
    final_decision = normalize_text(record.get("final_decision", ""))

    parts = []
    if question:
        parts.append(f"Question: {question}")
    if contexts:
        parts.append("Context: " + " ".join(contexts))
    if long_answer:
        parts.append(f"Explanation: {long_answer}")
    if final_decision:
        parts.append(f"Answer: {final_decision}")
    return "\n".join(parts).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-txt", default="")
    parser.add_argument("--min-chars", type=int, default=120)
    args = parser.parse_args()

    response = requests.get(args.source_url, timeout=120)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for pmid, record in payload.items():
        text = build_text(record)
        if len(text) < args.min_chars:
            continue
        rows.append(
            {
                "source": "pubmedqa_pqal",
                "doc_id": str(pmid),
                "title": normalize_text(record.get("QUESTION", ""))[:200],
                "url": args.source_url,
                "text": text,
                "char_count": len(text),
            }
        )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output_txt:
        output_txt = Path(args.output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text("\n\n".join(row["text"] for row in rows), encoding="utf-8")

    print(
        json.dumps(
            {
                "source_url": args.source_url,
                "kept_documents": len(rows),
                "output_jsonl": str(output_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
