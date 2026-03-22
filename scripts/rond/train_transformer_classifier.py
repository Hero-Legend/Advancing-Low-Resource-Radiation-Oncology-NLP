import argparse
import inspect
import json
from collections import Counter
from pathlib import Path
import random

from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, Trainer, TrainingArguments


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_dataset(rows, label_to_id):
    return Dataset.from_list([
        {'text': row['text'], 'label': label_to_id[row['label']]}
        for row in rows
    ])


def oversample_rows(rows, seed):
    label_to_rows = {}
    for row in rows:
        label_to_rows.setdefault(row["label"], []).append(row)

    max_count = max(len(group) for group in label_to_rows.values())
    rng = random.Random(seed)
    oversampled = []
    for label, group in sorted(label_to_rows.items()):
        oversampled.extend(group)
        deficit = max_count - len(group)
        if deficit > 0:
            oversampled.extend(rng.choice(group) for _ in range(deficit))
    rng.shuffle(oversampled)
    return oversampled


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def build_prediction_rows(rows, prediction_output, id_to_label):
    logits = prediction_output.predictions
    label_ids = prediction_output.label_ids
    pred_ids = logits.argmax(axis=-1)
    probs = softmax(logits)
    prediction_rows = []
    for row, true_id, pred_id, prob_vec in zip(rows, label_ids, pred_ids, probs):
        prediction_rows.append(
            {
                "text": row["text"],
                "true_label": id_to_label[int(true_id)],
                "pred_label": id_to_label[int(pred_id)],
                "true_id": int(true_id),
                "pred_id": int(pred_id),
                "probabilities": {
                    id_to_label[idx]: float(prob_vec[idx]) for idx in range(len(prob_vec))
                },
            }
        )
    return prediction_rows


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in kwargs and "processing_class" not in trainer_params:
            kwargs["tokenizer"] = kwargs.pop("processing_class")
        if "tokenizer" in kwargs and "tokenizer" not in trainer_params and "processing_class" in trainer_params:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
        )
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-class-weights', action='store_true')
    parser.add_argument('--oversample-train', action='store_true')
    parser.add_argument('--disable-checkpointing', action='store_true')
    parser.add_argument('--skip-final-save', action='store_true')
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    test_rows = load_jsonl(args.test)
    if args.oversample_train:
        train_rows = oversample_rows(train_rows, args.seed)
    labels = sorted({row['label'] for row in train_rows + val_rows + test_rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    label_counts = Counter(row['label'] for row in train_rows)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except ValueError as exc:
        # Older local checkpoints such as BioBERT may only ship a vocab file.
        # In that case, fall back to the slow tokenizer instead of failing.
        if "backend tokenizer" not in str(exc):
            raise
        vocab_path = Path(args.model_name) / "vocab.txt"
        tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=args.max_length)

    train_ds = build_dataset(train_rows, label_to_id).map(tokenize, batched=True)
    val_ds = build_dataset(val_rows, label_to_id).map(tokenize, batched=True)
    test_ds = build_dataset(test_rows, label_to_id).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            'accuracy': accuracy_score(labels_arr, preds),
            'macro_f1': f1_score(labels_arr, preds, average='macro'),
            'weighted_f1': f1_score(labels_arr, preds, average='weighted'),
        }

    training_kwargs = {
        'output_dir': args.output_dir,
        'learning_rate': args.learning_rate,
        'per_device_train_batch_size': args.train_batch_size,
        'per_device_eval_batch_size': args.eval_batch_size,
        'num_train_epochs': args.epochs,
        'eval_strategy': 'epoch',
        'seed': args.seed,
        'logging_dir': str(Path(args.output_dir) / 'logs'),
        'report_to': 'none',
    }
    if args.disable_checkpointing:
        training_kwargs.update(
            {
                'save_strategy': 'no',
                'load_best_model_at_end': False,
            }
        )
    else:
        training_kwargs.update(
            {
                'save_strategy': 'epoch',
                'load_best_model_at_end': True,
                'metric_for_best_model': 'macro_f1',
                'greater_is_better': True,
            }
        )
    training_args = TrainingArguments(**training_kwargs)

    class_weights = None
    if args.use_class_weights:
        total_count = sum(label_counts.values())
        weights = []
        for label in labels:
            count = label_counts[label]
            weights.append(total_count / (len(labels) * count))
        class_weights = torch.tensor(weights, dtype=torch.float)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix='test')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_final_save:
        trainer.save_model(str(out_dir / "final_model"))
        tokenizer.save_pretrained(str(out_dir / "final_model"))

    val_predictions = trainer.predict(val_ds)
    test_predictions = trainer.predict(test_ds)

    with open(out_dir / 'label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, indent=2)
    with open(out_dir / 'train_label_distribution.json', 'w', encoding='utf-8') as f:
        json.dump(label_counts, f, indent=2)
    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'model_name': args.model_name,
                'max_length': args.max_length,
                'learning_rate': args.learning_rate,
                'train_batch_size': args.train_batch_size,
                'eval_batch_size': args.eval_batch_size,
                'epochs': args.epochs,
                'seed': args.seed,
                'use_class_weights': args.use_class_weights,
                'oversample_train': args.oversample_train,
                'disable_checkpointing': args.disable_checkpointing,
                'skip_final_save': args.skip_final_save,
            },
            f,
            indent=2,
        )
    with open(out_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'val': val_metrics, 'test': test_metrics}, f, indent=2)
    with open(out_dir / 'val_predictions.jsonl', 'w', encoding='utf-8') as f:
        for row in build_prediction_rows(val_rows, val_predictions, id_to_label):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(out_dir / 'test_predictions.jsonl', 'w', encoding='utf-8') as f:
        for row in build_prediction_rows(test_rows, test_predictions, id_to_label):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
