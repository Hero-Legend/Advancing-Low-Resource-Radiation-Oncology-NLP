import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_rond_texts(root):
    root_path = Path(root)
    for csv_path in root_path.rglob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    text_parts = []
                    for key, value in row.items():
                        key_lower = str(key).lower()
                        if any(field in key_lower for field in ["text", "question", "summary", "prompt", "instruction", "input"]):
                            if value and str(value).strip():
                                text_parts.append(str(value).strip())
                    text = "\n".join(text_parts).strip()
                    if text:
                        yield {
                            "source": "rond_anchor",
                            "doc_id": f"{csv_path.stem}_{idx}",
                            "title": csv_path.stem,
                            "url": "",
                            "text": text,
                        }
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmc-jsonl", default="")
    parser.add_argument("--nci-jsonl", default="")
    parser.add_argument("--rond-root", default="")
    parser.add_argument("--include-rond-anchor", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    docs = []
    if args.pmc_jsonl:
        docs.extend(iter_jsonl(args.pmc_jsonl))
    if args.nci_jsonl:
        docs.extend(iter_jsonl(args.nci_jsonl))
    if args.include_rond_anchor and args.rond_root:
        docs.extend(iter_rond_texts(args.rond_root))

    seen = set()
    kept = []
    for doc in docs:
        text = normalize_text(doc.get("text", ""))
        if len(text) < args.min_chars:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(
            {
                "source": doc.get("source", ""),
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "text": text,
                "char_count": len(text),
            }
        )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for doc in kept:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    output_txt = Path(args.output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n\n".join(doc["text"] for doc in kept), encoding="utf-8")

    print(json.dumps({"input_documents": len(docs), "kept_documents": len(kept)}, indent=2))


if __name__ == "__main__":
    main()
