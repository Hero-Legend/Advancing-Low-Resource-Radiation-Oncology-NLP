import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    print("RUN", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True, cwd=str(cwd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--keywords-file", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--window-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--adaptation-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--skip-adaptation", action="store_true")
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    corpora_root = output_root / "corpora"
    stats_root = output_root / "stats"
    adaptation_root = output_root / "adaptation_models"
    downstream_root = output_root / "downstream"
    config_root = output_root / "configs"
    for path in [corpora_root, stats_root, adaptation_root, downstream_root, config_root]:
        path.mkdir(parents=True, exist_ok=True)

    build_script = project_root / "scripts" / "public_corpus" / "build_keyword_focused_corpus.py"
    stats_script = project_root / "scripts" / "public_corpus" / "compute_corpus_stats.py"
    mlm_script = project_root / "scripts" / "domain_adaptation" / "train_mlm_adaptation.py"
    multiseed_script = project_root / "scripts" / "rond" / "run_multiseed_experiments.py"

    manifest = {
        "raw_jsonl": args.raw_jsonl,
        "keywords_file": args.keywords_file,
        "tokenizer_model": args.tokenizer_model,
        "base_model": args.base_model,
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "window_sizes": args.window_sizes,
        "seeds": args.seeds,
        "adaptation_epochs": args.adaptation_epochs,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "skip_adaptation": args.skip_adaptation,
        "skip_downstream": args.skip_downstream,
    }
    (output_root / "window_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for window_size in args.window_sizes:
        tag = f"w{window_size}"
        focused_jsonl = corpora_root / f"public_radonc_adaptation_focused_{tag}.jsonl"
        focused_txt = corpora_root / f"public_radonc_adaptation_focused_{tag}.txt"
        stats_json = stats_root / f"public_radonc_adaptation_focused_{tag}_stats.json"
        model_dir = adaptation_root / f"pubmedbert_public_radonc_focused_{tag}_mlm_e{args.adaptation_epochs}"
        downstream_dir = downstream_root / f"window_{tag}"
        config_json = config_root / f"window_{tag}_configs.json"

        run_cmd(
            [
                sys.executable,
                build_script,
                "--input-jsonl",
                args.raw_jsonl,
                "--keywords-file",
                args.keywords_file,
                "--output-jsonl",
                focused_jsonl,
                "--output-txt",
                focused_txt,
                "--window-size",
                str(window_size),
            ],
            project_root,
        )

        run_cmd(
            [
                sys.executable,
                stats_script,
                "--input-jsonl",
                focused_jsonl,
                "--tokenizer",
                args.tokenizer_model,
                "--output-json",
                stats_json,
            ],
            project_root,
        )

        if not args.skip_adaptation:
            run_cmd(
                [
                    sys.executable,
                    mlm_script,
                    "--train-jsonl",
                    focused_jsonl,
                    "--model-name",
                    args.base_model,
                    "--output-dir",
                    model_dir,
                    "--epochs",
                    str(args.adaptation_epochs),
                    "--train-batch-size",
                    str(args.train_batch_size),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--max-seq-length",
                    str(args.max_seq_length),
                ],
                project_root,
            )

        configs = [
            {
                "name": f"pubmedbert_focused_{tag}_weighted",
                "model_name": str(model_dir),
                "use_class_weights": True,
            }
        ]
        config_json.write_text(json.dumps(configs, indent=2), encoding="utf-8")

        if not args.skip_downstream:
            run_cmd(
                [
                    sys.executable,
                    multiseed_script,
                    "--train",
                    args.train,
                    "--val",
                    args.val,
                    "--test",
                    args.test,
                    "--output-root",
                    downstream_dir,
                    "--configs-json",
                    config_json,
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                    "--cwd",
                    project_root,
                ],
                project_root,
            )


if __name__ == "__main__":
    main()
