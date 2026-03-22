import argparse
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def collect_metrics(root):
    records = []
    for metrics_path in sorted(Path(root).glob("*/seed_*/metrics_summary.json")):
        config_name = metrics_path.parent.parent.name
        seed_name = metrics_path.parent.name
        seed = int(seed_name.split("_", 1)[1])
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        records.append(
            {
                "config_name": config_name,
                "seed": seed,
                "val_macro_f1": payload["val"]["eval_macro_f1"],
                "test_macro_f1": payload["test"]["test_macro_f1"],
                "val_accuracy": payload["val"]["eval_accuracy"],
                "test_accuracy": payload["test"]["test_accuracy"],
                "test_weighted_f1": payload["test"]["test_weighted_f1"],
            }
        )
    return records


def summarize(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["config_name"], []).append(record)

    summary = []
    for config_name, items in grouped.items():
        summary.append(
            {
                "config_name": config_name,
                "runs": len(items),
                "val_macro_f1_mean": mean([item["val_macro_f1"] for item in items]),
                "val_macro_f1_std": std([item["val_macro_f1"] for item in items]),
                "test_macro_f1_mean": mean([item["test_macro_f1"] for item in items]),
                "test_macro_f1_std": std([item["test_macro_f1"] for item in items]),
                "val_accuracy_mean": mean([item["val_accuracy"] for item in items]),
                "test_accuracy_mean": mean([item["test_accuracy"] for item in items]),
                "test_weighted_f1_mean": mean([item["test_weighted_f1"] for item in items]),
            }
        )
    summary.sort(key=lambda item: item["test_macro_f1_mean"], reverse=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    records = collect_metrics(args.results_root)
    summary = summarize(records)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"records": records, "summary": summary}, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "config_name,runs,val_macro_f1_mean,val_macro_f1_std,test_macro_f1_mean,test_macro_f1_std,val_accuracy_mean,test_accuracy_mean,test_weighted_f1_mean"
    ]
    for item in summary:
        lines.append(
            ",".join(
                [
                    item["config_name"],
                    str(item["runs"]),
                    f"{item['val_macro_f1_mean']:.6f}",
                    f"{item['val_macro_f1_std']:.6f}",
                    f"{item['test_macro_f1_mean']:.6f}",
                    f"{item['test_macro_f1_std']:.6f}",
                    f"{item['val_accuracy_mean']:.6f}",
                    f"{item['test_accuracy_mean']:.6f}",
                    f"{item['test_weighted_f1_mean']:.6f}",
                ]
            )
        )
    output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
