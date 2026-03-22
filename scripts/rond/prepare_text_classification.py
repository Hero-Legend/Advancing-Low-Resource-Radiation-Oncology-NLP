import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def normalize_label(label):
    return label.strip().rstrip('.')


def stratified_split(rows, label_key, seed, train_ratio, val_ratio):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[label_key]].append(row)

    train, val, test = [], [], []
    for _, items in buckets.items():
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
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue
            text = row[0].strip()
            label = normalize_label(row[1])
            rows.append({'id': idx, 'text': text, 'label': label})

    train, val, test = stratified_split(rows, 'label', args.seed, args.train_ratio, args.val_ratio)

    write_jsonl(output_dir / 'train.jsonl', train)
    write_jsonl(output_dir / 'val.jsonl', val)
    write_jsonl(output_dir / 'test.jsonl', test)

    metadata = {
        'source_file': str(input_path),
        'num_examples': len(rows),
        'labels': sorted({row['label'] for row in rows}),
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'seed': args.seed,
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
