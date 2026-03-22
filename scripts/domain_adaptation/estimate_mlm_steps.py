import argparse
import json
import math
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    total_sequences = 0
    total_raw_tokens = 0
    max_sequences_per_doc = 0

    for row in dataset:
        text = row.get(args.text_field, "")
        raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        total_raw_tokens += len(raw_ids)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            return_overflowing_tokens=True,
        )
        sequence_count = len(encoded["input_ids"])
        total_sequences += sequence_count
        max_sequences_per_doc = max(max_sequences_per_doc, sequence_count)

    optimizer_steps_per_epoch = math.ceil(total_sequences / args.train_batch_size)
    payload = {
        "train_jsonl": args.train_jsonl,
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "document_count": len(dataset),
        "raw_token_count": total_raw_tokens,
        "overflow_sequence_count": total_sequences,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "max_sequences_per_document": max_sequences_per_doc,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
