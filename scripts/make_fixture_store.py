"""Write a synthetic data/ and mlruns/ under a directory, for the container smoke test.

usage: python scripts/make_fixture_store.py DIR
"""

import sys
from pathlib import Path

from rail_risk.fixtures import build_fixture_store

if __name__ == "__main__":
    root = Path(sys.argv[1])
    print(build_fixture_store(root))
