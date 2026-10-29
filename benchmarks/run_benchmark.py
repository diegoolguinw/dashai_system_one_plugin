#!/usr/bin/env python3
"""Run every benchmark configuration, one subprocess each, and store JSON results."""

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# (model, languages, devices). English-only models are not run on Spanish.
CONFIGS = [
    ("laya", ["en"], ["cpu", "mps"]),
    ("laya-multilingual", ["en", "es"], ["cpu", "mps"]),
    ("gliner", ["en"], ["cpu", "mps"]),
    ("gliner-multilingual", ["en", "es"], ["cpu", "mps"]),
    ("kev", ["en"], ["mps"]),  # served by MLX on Apple Silicon; the device flag is not used
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="interpreter with laya, gliner2 and the plugin")
    ap.add_argument("--data-dir", type=pathlib.Path, default=HERE / "data" / "test")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--only", help="comma separated model names to run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    for model, langs, devices in CONFIGS:
        if only and model not in only:
            continue
        for device in devices:
            for lang in langs:
                target = args.out / f"{model}__{device}__{lang}.json"
                if target.exists():
                    print(f"skip {target.name} (exists)")
                    continue
                print(f"run  {target.name}", flush=True)
                cmd = [
                    args.python, "-I", str(HERE / "worker.py"), "--model", model,
                    "--device", device, "--lang", lang, "--n", str(args.n),
                    "--data", str(args.data_dir / f"{lang}.json.gz"),
                ]
                env = {"PYTHONPATH": str(ROOT / "src")}
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, check=False, env={**os.environ, **env}
                )
                if proc.returncode != 0:
                    print(proc.stderr[-2000:], file=sys.stderr)
                    continue
                target.write_text(proc.stdout.strip().splitlines()[-1] + "\n")


if __name__ == "__main__":
    main()
