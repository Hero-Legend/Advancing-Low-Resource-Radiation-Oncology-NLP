import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--configs-json", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--trainer-script", default="scripts/rond/train_transformer_classifier.py")
    parser.add_argument("--cwd", default="")
    args = parser.parse_args()

    configs = json.loads(Path(args.configs_json).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trainer_script = Path(args.trainer_script).resolve()
    run_cwd = str(Path(args.cwd).resolve()) if args.cwd else str(Path.cwd())

    run_manifest = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "seeds": args.seeds,
        "configs": configs,
        "trainer_script": str(trainer_script),
        "cwd": run_cwd,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    for config in configs:
        config_name = config["name"]
        model_name = config["model_name"]
        use_class_weights = config.get("use_class_weights", False)
        extra_args = config.get("extra_args", [])

        for seed in args.seeds:
            run_dir = output_root / config_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(trainer_script),
                "--train",
                args.train,
                "--val",
                args.val,
                "--test",
                args.test,
                "--model-name",
                model_name,
                "--output-dir",
                str(run_dir),
                "--seed",
                str(seed),
            ]
            if use_class_weights:
                cmd.append("--use-class-weights")
            cmd.extend(extra_args)

            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=run_cwd)


if __name__ == "__main__":
    main()
