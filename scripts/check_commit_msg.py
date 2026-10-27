#!/usr/bin/env python3
"""commit-msg hook: reject messages with an AI trailer or tool credit.

python3 scripts/check_commit_msg.py .git/COMMIT_EDITMSG
"""

import re
import sys

# built from pieces so these words do not appear in the file itself
BANNED = re.compile(
    "|".join(("co-auth" + "ored-by", "cl" + "aude", "anthr" + "opic", "generated" + " with")),
    re.IGNORECASE,
)


def main(path: str) -> int:
    text = open(path, encoding="utf-8").read()
    hit = BANNED.search(text)
    if hit:
        print(f"commit message rejected: contains {hit.group(0)!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
