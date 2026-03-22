import argparse
import csv
import json
import math
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_bundle(rows, labels):
    class_rows = []
    f1_values = []
    weighted_num = 0.0

    for label in labels:
        support = sum(1 for row in rows if row["true_label"] == label)
        tp = sum(1 for row in rows if row["true_label"] == label and row["pred_label"] == label)
        fp = sum(1 for row in rows if row["true_label"] != label and row["pred_label"] == label)
        fn = sum(1 for row in rows if row["true_label"] == label and row["pred_label"] != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        weighted_num += support * f1

        class_rows.append(
            {
                "label": label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    accuracy = sum(1 for row in rows if row["true_label"] == row["pred_label"]) / len(rows) if rows else 0.0
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    weighted_f1 = weighted_num / len(rows) if rows else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_rows": class_rows,
    }


def percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    idx = int(math.floor((len(sorted_values) - 1) * q))
    return float(sorted_values[idx])


def bootstrap_ci(rows, labels, iterations=2000, seed=42):
    import random

    rng = random.Random(seed)
    acc_values = []
    macro_values = []
    for _ in range(iterations):
        sampled = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        metrics = metric_bundle(sampled, labels)
        acc_values.append(metrics["accuracy"])
        macro_values.append(metrics["macro_f1"])
    acc_values.sort()
    macro_values.sort()
    return {
        "accuracy_ci_lower": percentile(acc_values, 0.025),
        "accuracy_ci_upper": percentile(acc_values, 0.975),
        "macro_f1_ci_lower": percentile(macro_values, 0.025),
        "macro_f1_ci_upper": percentile(macro_values, 0.975),
    }


def log_binom_coeff(n, k):
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    value = 0.0
    for i in range(1, k + 1):
        value += math.log(n - k + i) - math.log(i)
    return value


def exact_sign_test_pvalue(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    smaller = min(wins, losses)
    probs = []
    for k in range(0, smaller + 1):
        logp = log_binom_coeff(n, k) - n * math.log(2.0)
        probs.append(math.exp(logp))
    p = min(1.0, 2.0 * sum(probs))
    return p


def pairwise_summary(reference_rows, comparison_rows):
    ref_by_text = {row["text"]: row for row in reference_rows}
    cmp_by_text = {row["text"]: row for row in comparison_rows}

    wins = 0
    losses = 0
    ties = 0
    for text, ref_row in ref_by_text.items():
        cmp_row = cmp_by_text[text]
        ref_correct = ref_row["true_label"] == ref_row["pred_label"]
        cmp_correct = cmp_row["true_label"] == cmp_row["pred_label"]
        if ref_correct and not cmp_correct:
            wins += 1
        elif cmp_correct and not ref_correct:
            losses += 1
        else:
            ties += 1

    return {
        "reference_wins": wins,
        "comparison_wins": losses,
        "ties": ties,
        "p_value": exact_sign_test_pvalue(wins, losses),
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-json", required=True, help="JSON file: [{'system':..., 'predictions':...}, ...]")
    parser.add_argument("--reference-system", required=True)
    parser.add_argument("--bootstrap-csv", required=True)
    parser.add_argument("--pairwise-csv", required=True)
    parser.add_argument("--classwise-csv", required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    systems = json.loads(Path(args.systems_json).read_text(encoding="utf-8"))
    loaded = []
    all_labels = set()
    for item in systems:
        rows = load_jsonl(Path(item["predictions"]))
        loaded.append({"system": item["system"], "rows": rows})
        for row in rows:
            all_labels.add(row["true_label"])
            all_labels.add(row["pred_label"])
    labels = sorted(all_labels)

    bootstrap_rows = []
    classwise_rows = []
    for item in loaded:
        metrics = metric_bundle(item["rows"], labels)
        ci = bootstrap_ci(item["rows"], labels, iterations=args.iterations, seed=args.seed)
        bootstrap_rows.append(
            {
                "system": item["system"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy_ci_lower": ci["accuracy_ci_lower"],
                "accuracy_ci_upper": ci["accuracy_ci_upper"],
                "macro_f1_ci_lower": ci["macro_f1_ci_lower"],
                "macro_f1_ci_upper": ci["macro_f1_ci_upper"],
            }
        )
        for class_row in metrics["class_rows"]:
            classwise_rows.append(
                {
                    "system": item["system"],
                    **class_row,
                }
            )

    ref_rows = next(item["rows"] for item in loaded if item["system"] == args.reference_system)
    pairwise_rows = []
    for item in loaded:
        if item["system"] == args.reference_system:
            continue
        summary = pairwise_summary(ref_rows, item["rows"])
        pairwise_rows.append(
            {
                "reference_system": args.reference_system,
                "comparison_system": item["system"],
                **summary,
            }
        )

    write_csv(
        Path(args.bootstrap_csv),
        bootstrap_rows,
        [
            "system",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "accuracy_ci_lower",
            "accuracy_ci_upper",
            "macro_f1_ci_lower",
            "macro_f1_ci_upper",
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
            "p_value",
        ],
    )
    write_csv(
        Path(args.classwise_csv),
        classwise_rows,
        ["system", "label", "support", "tp", "fp", "fn", "precision", "recall", "f1"],
    )

    print(f"Wrote {args.bootstrap_csv}")
    print(f"Wrote {args.pairwise_csv}")
    print(f"Wrote {args.classwise_csv}")


if __name__ == "__main__":
    main()
