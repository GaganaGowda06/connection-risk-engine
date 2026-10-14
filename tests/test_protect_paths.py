import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).parent.parent / ".tooling" / "hooks" / "protect_paths.py"


def run_hook(payload: dict | str) -> subprocess.CompletedProcess:
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(  # noqa: S603
        [sys.executable, str(HOOK)], input=stdin, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "/abs/project/.env",
        "sub/.env",
        "data/raw/x.csv",
        "/abs/project/data/raw/2024/file.json",
        "./data/raw/x",
        "data/raw/../raw/x",
    ],
)
def test_blocked(path):
    r = run_hook({"tool_name": "Write", "tool_input": {"file_path": path}})
    assert r.returncode == 2
    assert path in r.stderr


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "raw/data.txt",
        "data/processed/x.parquet",
        "data/.gitkeep",
        "src/rail_risk/config.py",
        "data/raw",
        "data/rawish/x",
        "env",
    ],
)
def test_allowed(path):
    assert run_hook({"tool_name": "Edit", "tool_input": {"file_path": path}}).returncode == 0


def test_notebook_path_blocked():
    r = run_hook({"tool_name": "NotebookEdit", "tool_input": {"notebook_path": "data/raw/a.ipynb"}})
    assert r.returncode == 2


@pytest.mark.parametrize("stdin", ["not json", "{}", '{"tool_input": null}'])
def test_unparsable_or_empty_allowed(stdin):
    assert run_hook(stdin).returncode == 0
