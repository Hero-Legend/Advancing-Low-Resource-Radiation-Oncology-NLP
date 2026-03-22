import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(batch):
        tokenized = tokenizer(
            batch[args.text_field],
            truncation=True,
            max_length=args.max_seq_length,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
        )
        tokenized.pop("overflow_to_sample_mapping", None)
        return tokenized

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing adaptation corpus",
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "adaptation_run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_jsonl": args.train_jsonl,
                "model_name": args.model_name,
                "text_field": args.text_field,
                "max_seq_length": args.max_seq_length,
                "learning_rate": args.learning_rate,
                "train_batch_size": args.train_batch_size,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "mlm_probability": args.mlm_probability,
                "seed": args.seed,
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
