#!/usr/bin/env python3
"""Download the MASSIVE scenario test splits used by the benchmark (pinned revision)."""

import argparse
import pathlib

from huggingface_hub import hf_hub_download

REPO = "mteb/amazon_massive_scenario"  # MASSIVE (CC-BY-4.0): parallel utterances per language
REVISION = "58871793b91addb7c5f7afff26ccf08737fb6697"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("benchmarks/data"))
    args = parser.parse_args()
    for lang in ("en", "es"):
        path = hf_hub_download(
            REPO, f"test/{lang}.json.gz", repo_type="dataset",
            revision=REVISION, local_dir=args.out,
        )
        print(path)


if __name__ == "__main__":
    main()
