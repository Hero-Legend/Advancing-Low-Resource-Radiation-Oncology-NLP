import argparse
import json
import re
from pathlib import Path


def load_keywords(path):
    return [
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def keep_windows(sentences, keywords, window_size):
    keep_indices = set()
    for idx, sentence in enumerate(sentences):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            start = max(0, idx - window_size)
            end = min(len(sentences), idx + window_size + 1)
            keep_indices.update(range(start, end))
    return [sentences[idx] for idx in sorted(keep_indices)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--keywords-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    keywords = load_keywords(args.keywords_file)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_txt = Path(args.output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sentences = split_sentences(row.get("text", ""))
            focused_sentences = keep_windows(sentences, keywords, args.window_size)
            focused_text = " ".join(focused_sentences).strip()
            if len(focused_text) < args.min_chars:
                continue
            kept.append(
                {
                    "source": f"{row.get('source', '')}_focused",
                    "doc_id": row.get("doc_id", ""),
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "text": focused_text,
                    "char_count": len(focused_text),
                }
            )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_txt.write_text("\n\n".join(row["text"] for row in kept), encoding="utf-8")
    print(json.dumps({"kept_documents": len(kept)}, indent=2))


if __name__ == "__main__":
    main()
