#!/usr/bin/env python3
"""Run the FinReason tests and execute the main notebook."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "FinReason_Implementation.ipynb"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def check_environment() -> None:
    if importlib.util.find_spec("finreason") is None:
        raise RuntimeError(
            "FinReason is not installed. Run "
            f"'{sys.executable} -m pip install -r requirements.txt' first."
        )
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")


def run_tests() -> None:
    print("\nRunning unit tests...", flush=True)
    run([
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ])


def notebook_environment() -> dict[str, str]:
    jupyter_path = PROJECT_ROOT / "workspace" / "cache" / "jupyter"
    kernel_dir = jupyter_path / "kernels" / "finreason"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3 (FinReason)",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(
        json.dumps(kernel_spec, indent=2) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    if env.get("JUPYTER_PATH"):
        env["JUPYTER_PATH"] = str(jupyter_path) + os.pathsep + env["JUPYTER_PATH"]
    else:
        env["JUPYTER_PATH"] = str(jupyter_path)
    return env


def execute_notebook() -> None:
    print("\nExecuting the notebook...", flush=True)
    run([
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.kernel_name=finreason",
        "--ExecutePreprocessor.timeout=-1",
        str(NOTEBOOK),
    ], env=notebook_environment())

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    unexecuted = [cell for cell in code_cells if cell.get("execution_count") is None]

    if errors:
        names = ", ".join(error.get("ename", "unknown error") for error in errors)
        raise RuntimeError(f"Notebook contains execution errors: {names}")
    if unexecuted:
        raise RuntimeError(f"Notebook has {len(unexecuted)} unexecuted code cell(s)")

    print(f"Notebook complete: {len(code_cells)} code cells executed with no errors.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tests-only", action="store_true", help="run the unit tests and stop")
    mode.add_argument("--notebook-only", action="store_true", help="execute only the notebook")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_environment()
    if not args.notebook_only:
        run_tests()
    if not args.tests_only:
        execute_notebook()
    print("\nFinReason completed successfully.")


if __name__ == "__main__":
    main()
