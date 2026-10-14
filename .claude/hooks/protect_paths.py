#!/usr/bin/env python3
"""PreToolUse hook: exit 2 for edits to .env or anything inside data/raw/."""

import json
import os
import sys
from pathlib import PurePosixPath


def is_protected(file_path: str) -> bool:
    parts = PurePosixPath(os.path.normpath(file_path.replace("\\", "/"))).parts
    if parts and parts[-1] == ".env":
        return True
    return any(
        parts[i] == "data" and parts[i + 1] == "raw" and i + 2 < len(parts)
        for i in range(len(parts) - 1)
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if isinstance(path, str) and is_protected(path):
        print(
            f"Blocked: {path} is protected (.env and data/raw are never edited).", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
