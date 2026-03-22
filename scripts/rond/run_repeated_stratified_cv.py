import argparse
import json
import subprocess
import sys
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_from_cwd(path_str: str, cwd: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--configs-json", required=True)
    parser.add_argument("--trainer-script", default="scripts/rond/train_transformer_classifier.py")
    parser.add_argument("--cwd", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.125)
    args = parser.parse_args()

    run_cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve()
    train_path = resolve_from_cwd(args.train, run_cwd)
    val_path = resolve_from_cwd(args.val, run_cwd)
    test_path = resolve_from_cwd(args.test, run_cwd)
    output_root = resolve_from_cwd(args.output_root, run_cwd)
    configs_path = resolve_from_cwd(args.configs_json, run_cwd)
    trainer_script = resolve_from_cwd(args.trainer_script, run_cwd)

    configs = json.loads(configs_path.read_text(encoding="utf-8"))
    all_rows = load_jsonl(train_path) + load_jsonl(val_path) + load_jsonl(test_path)
    labels = [row["label"] for row in all_rows]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "configs": configs,
        "trainer_script": str(trainer_script),
        "cwd": str(run_cwd),
        "total_rows": len(all_rows),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )

    split_records = []
    for split_index, (train_val_idx, test_idx) in enumerate(splitter.split(all_rows, labels)):
        repeat_idx = split_index // args.folds
        fold_idx = split_index % args.folds
        split_name = f"repeat_{repeat_idx}_fold_{fold_idx}"

        train_val_rows = [all_rows[idx] for idx in train_val_idx]
        train_val_labels = [labels[idx] for idx in train_val_idx]
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.val_fraction,
            random_state=args.seed + split_index,
        )
        inner_train_idx, inner_val_idx = next(sss.split(train_val_rows, train_val_labels))

        train_rows = [train_val_rows[idx] for idx in inner_train_idx]
        val_rows = [train_val_rows[idx] for idx in inner_val_idx]
        test_rows = [all_rows[idx] for idx in test_idx]

        split_dir = output_root / "splits" / split_name
        train_file = split_dir / "train.jsonl"
        val_file = split_dir / "val.jsonl"
        test_file = split_dir / "test.jsonl"
        write_jsonl(train_file, train_rows)
        write_jsonl(val_file, val_rows)
        write_jsonl(test_file, test_rows)

        split_record = {
            "split_name": split_name,
            "repeat_idx": repeat_idx,
            "fold_idx": fold_idx,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_size": len(test_rows),
        }
        split_records.append(split_record)

        for config in configs:
            run_dir = output_root / "runs" / config["name"] / split_name
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(trainer_script),
                "--train",
                str(train_file),
                "--val",
                str(val_file),
                "--test",
                str(test_file),
                "--model-name",
                config["model_name"],
                "--output-dir",
                str(run_dir),
                "--seed",
                str(args.seed + split_index),
            ]
            if config.get("use_class_weights", False):
                cmd.append("--use-class-weights")
            cmd.extend(config.get("extra_args", []))

            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=str(run_cwd))

    (output_root / "split_manifest.json").write_text(json.dumps(split_records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
