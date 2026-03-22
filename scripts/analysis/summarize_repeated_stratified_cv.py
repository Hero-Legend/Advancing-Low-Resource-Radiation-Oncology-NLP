import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--per-fold-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--pairwise-csv", required=True)
    parser.add_argument("--reference-system", required=True)
    args = parser.parse_args()

    cv_root = Path(args.cv_root)
    metric_paths = sorted(cv_root.glob("runs/*/repeat_*_fold_*/metrics_summary.json"))

    per_fold_rows = []
    for path in metric_paths:
        config_name = path.parent.parent.name
        split_name = path.parent.name
        metrics = json.loads(path.read_text(encoding="utf-8"))
        test_metrics = metrics["test"]
        repeat_part, fold_part = split_name.split("_fold_")
        repeat_idx = int(repeat_part.replace("repeat_", ""))
        fold_idx = int(fold_part)
        per_fold_rows.append(
            {
                "system": config_name,
                "split_name": split_name,
                "repeat_idx": repeat_idx,
                "fold_idx": fold_idx,
                "test_accuracy": test_metrics["test_accuracy"],
                "test_macro_f1": test_metrics["test_macro_f1"],
                "test_weighted_f1": test_metrics["test_weighted_f1"],
            }
        )

    systems = sorted({row["system"] for row in per_fold_rows})
    summary_rows = []
    for system in systems:
        rows = [row for row in per_fold_rows if row["system"] == system]
        accs = [row["test_accuracy"] for row in rows]
        macros = [row["test_macro_f1"] for row in rows]
        weighted = [row["test_weighted_f1"] for row in rows]
        summary_rows.append(
            {
                "system": system,
                "num_splits": len(rows),
                "mean_test_accuracy": mean(accs),
                "std_test_accuracy": safe_std(accs),
                "mean_test_macro_f1": mean(macros),
                "std_test_macro_f1": safe_std(macros),
                "mean_test_weighted_f1": mean(weighted),
                "std_test_weighted_f1": safe_std(weighted),
            }
        )

    ref_rows = {row["split_name"]: row for row in per_fold_rows if row["system"] == args.reference_system}
    pairwise_rows = []
    for system in systems:
        if system == args.reference_system:
            continue
        cmp_rows = {row["split_name"]: row for row in per_fold_rows if row["system"] == system}
        wins = 0
        losses = 0
        ties = 0
        deltas = []
        for split_name, ref_row in ref_rows.items():
            cmp_row = cmp_rows[split_name]
            delta = ref_row["test_macro_f1"] - cmp_row["test_macro_f1"]
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        pairwise_rows.append(
            {
                "reference_system": args.reference_system,
                "comparison_system": system,
                "reference_wins": wins,
                "comparison_wins": losses,
                "ties": ties,
                "mean_macro_f1_delta": mean(deltas),
            }
        )

    write_csv(
        Path(args.per_fold_csv),
        per_fold_rows,
        [
            "system",
            "split_name",
            "repeat_idx",
            "fold_idx",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
        ],
    )
    write_csv(
        Path(args.summary_csv),
        summary_rows,
        [
            "system",
            "num_splits",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_macro_f1",
            "std_test_macro_f1",
            "mean_test_weighted_f1",
            "std_test_weighted_f1",
        ],
    )
    write_csv(
        Path(args.pairwise_csv),
        pairwise_rows,
        [
            "reference_system",
            "comparison_system",
            "reference_wins",
            "comparison_wins",
            "ties",
            "mean_macro_f1_delta",
        ],
    )

    print(f"Wrote {args.per_fold_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.pairwise_csv}")


if __name__ == "__main__":
    main()
