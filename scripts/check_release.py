#!/usr/bin/env python3
"""Pre-release checks: build, validate metadata, and test the wheel in a clean environment.

Nothing is uploaded. Run from the repository root:  python scripts/check_release.py
Requires `build` and `twine` (pip install -e ".[dev]").
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_PLUGINS = {
    "GLiNERDecideTextClassifier",
    "GLiNERMultiDecideTextClassifier",
    "LayaMultilingualTextClassifier",
    "LayaTextClassifier",
}


def run(*cmd: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{result.stdout}{result.stderr}")
    return result.stdout


def check(condition: bool, message: str) -> None:
    print(("ok   " if condition else "FAIL ") + message)
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unreleased", action="store_true",
                        help="do not require a release date in the CHANGELOG entry")
    args = parser.parse_args()
    init = (ROOT / "src/dashai_system_one_plugin/__init__.py").read_text()
    version = re.search(r'__version__ = "(.+?)"', init).group(1)
    changelog = (ROOT / "CHANGELOG.md").read_text()
    check(f"## [{version}]" in changelog, f"CHANGELOG has an entry for {version}")
    heading = changelog.split(f"## [{version}]")[1].split("\n")[0]
    if args.allow_unreleased:
        print("skip CHANGELOG release-date check (--allow-unreleased)")
    else:
        check("Sin publicar" not in heading, "CHANGELOG entry carries a release date")

    with tempfile.TemporaryDirectory() as tmp:
        dist = Path(tmp) / "dist"
        run(sys.executable, "-m", "build", "--outdir", str(dist))
        files = sorted(dist.iterdir())
        wheel = next(f for f in files if f.suffix == ".whl")
        check(len(files) == 2, f"built {[f.name for f in files]}")
        check(version in wheel.name, f"wheel version matches __version__ ({version})")
        run(sys.executable, "-m", "twine", "check", *map(str, files))
        print("ok   twine check passed")

        names = zipfile.ZipFile(wheel).namelist()
        check(not any(n.startswith(("tests", "benchmarks", "docs")) for n in names),
              "wheel contains only the package")
        entry = zipfile.ZipFile(wheel).read(
            next(n for n in names if n.endswith("entry_points.txt"))).decode()
        plugins = set(re.findall(r"^(\w+) = ", entry, re.M))
        check(plugins == EXPECTED_PLUGINS, f"entry points: {sorted(plugins)}")

        env = Path(tmp) / "env"
        py = str(env / "bin" / "python")
        if shutil.which("uv"):  # also works with uv-managed interpreters that lack ensurepip
            run("uv", "venv", "-q", "--python", sys.executable, str(env))
            run("uv", "pip", "install", "-q", "--python", py, str(wheel), "pytest")
        else:
            venv.create(env, with_pip=True)
            run(py, "-m", "pip", "install", "-q", str(wheel), "pytest")
        out = run(py, "-I", "-c",
                  "import importlib.metadata as m, dashai_system_one_plugin as p;"
                  "[e.load() for e in m.entry_points(group='dashai.plugins')"
                  " if e.value.startswith('dashai_system_one_plugin')];print(p.__version__)")
        check(out.strip() == version, "clean install imports and every entry point loads")
        tests = Path(tmp) / "tests"
        run("cp", "-R", str(ROOT / "tests"), str(tests))
        result = run(py, "-m", "pytest", str(tests), "-q", "-o", "pythonpath=",
                     "-o", "addopts=-m 'not integration'", "-p", "no:cacheprovider", cwd=Path(tmp))
        check(" passed" in result, "unit tests pass against the installed wheel")
    print(f"\nAll checks passed for {version}. Nothing was uploaded.")


if __name__ == "__main__":
    main()
