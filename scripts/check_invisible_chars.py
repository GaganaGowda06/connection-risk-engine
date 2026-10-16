"""Fail if a file contains zero-width, bidirectional-control or other invisible characters.

Used as a pre-commit hook on Markdown. Ordinary newlines and tabs are fine. Prints each hit with
file, line, column and the Unicode name, and exits 1 if there is any.
"""

import sys
import unicodedata
from pathlib import Path

ALLOWED = {"\n", "\r", "\t"}
EXTRA = {" ", " ", " ", " ", "　", "ᅟ", "ᅠ", "ㅤ", "﻿"}


def invisible(ch: str) -> bool:
    """Format (Cf) and control (Cc) characters, private use, and look-alike spaces."""
    if ch in ALLOWED:
        return False
    return unicodedata.category(ch) in {"Cf", "Cc", "Co"} or ch in EXTRA


def scan(text: str) -> list[tuple[int, int, str]]:
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        for column, ch in enumerate(line, start=1):
            if invisible(ch):
                hits.append((number, column, f"U+{ord(ch):04X} {unicodedata.name(ch, 'unnamed')}"))
    return hits


def main(paths: list[str]) -> int:
    failed = False
    for name in paths:
        for line, column, what in scan(Path(name).read_text(encoding="utf-8")):
            print(f"{name}:{line}:{column}: invisible character {what}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
