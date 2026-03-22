import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_stats(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def main():
    parser = argparse.ArgumentParser(description="Summarize window-size ablation outputs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--per-seed-csv", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    downstream_root = output_root / "downstream"
    corpus_stats_root = output_root / "stats"

    rows = []
    per_seed_rows = []

    for window_dir in sorted(downstream_root.glob("window_w*")):
        window_name = window_dir.name
        window_size = int(window_name.replace("window_w", "", 1))
        metrics_paths = sorted(window_dir.glob("**/metrics_summary.json"))

        test_macro_values = []
        test_accuracy_values = []
        val_macro_values = []
        completed_seeds = []

        for metrics_path in metrics_paths:
            metrics = load_json(metrics_path)
            seed = metrics_path.parent.name.replace("seed_", "")
            test_macro = metrics["test"]["test_macro_f1"]
            test_acc = metrics["test"]["test_accuracy"]
            val_macro = metrics["val"]["eval_macro_f1"]

            test_macro_values.append(test_macro)
            test_accuracy_values.append(test_acc)
            val_macro_values.append(val_macro)
            completed_seeds.append(seed)

            per_seed_rows.append(
                {
                    "window_size": window_size,
                    "seed": seed,
                    "val_macro_f1": val_macro,
                    "test_macro_f1": test_macro,
                    "test_accuracy": test_acc,
                    "metrics_path": str(metrics_path).replace("\\", "/"),
                }
            )

        stats_path = corpus_stats_root / f"public_radonc_adaptation_focused_w{window_size}_stats.json"
        corpus_stats = load_json(stats_path) if stats_path.exists() else {}

        test_macro_mean, test_macro_std = safe_stats(test_macro_values)
        test_acc_mean, test_acc_std = safe_stats(test_accuracy_values)
        val_macro_mean, val_macro_std = safe_stats(val_macro_values)

        rows.append(
            {
                "window_size": window_size,
                "completed_seed_count": len(completed_seeds),
                "completed_seeds": ",".join(completed_seeds),
                "doc_count": corpus_stats.get("document_count"),
                "char_count": corpus_stats.get("character_count"),
                "token_count": corpus_stats.get("token_count"),
                "avg_tokens_per_doc": corpus_stats.get("avg_tokens_per_document"),
                "test_macro_f1_mean": test_macro_mean,
                "test_macro_f1_std": test_macro_std,
                "test_accuracy_mean": test_acc_mean,
                "test_accuracy_std": test_acc_std,
                "val_macro_f1_mean": val_macro_mean,
                "val_macro_f1_std": val_macro_std,
            }
        )

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_size",
                "completed_seed_count",
                "completed_seeds",
                "doc_count",
                "char_count",
                "token_count",
                "avg_tokens_per_doc",
                "test_macro_f1_mean",
                "test_macro_f1_std",
                "test_accuracy_mean",
                "test_accuracy_std",
                "val_macro_f1_mean",
                "val_macro_f1_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    per_seed_csv = Path(args.per_seed_csv)
    per_seed_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_size",
                "seed",
                "val_macro_f1",
                "test_macro_f1",
                "test_accuracy",
                "metrics_path",
            ],
        )
        writer.writeheader()
        writer.writerows(per_seed_rows)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_seed_csv}")


if __name__ == "__main__":
    main()
