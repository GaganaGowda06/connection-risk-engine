import importlib.util
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_invisible_chars", Path(__file__).parent.parent / "scripts" / "check_invisible_chars.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
main, scan = _module.main, _module.scan

BAD = {
    "zero-width space": "a​b",
    "zero-width joiner": "a‍b",
    "bidi override": "a‮b",
    "bidi isolate": "a⁦b",
    "byte order mark": "﻿abc",
    "soft hyphen": "a­b",
    "word joiner": "a⁠b",
    "no-break space": "a b",
    "control character": "a\u0007b",
}


def test_every_kind_of_invisible_character_is_found_with_its_position():
    for label, text in BAD.items():
        hits = scan(f"line one\n{text}\n")
        assert hits and hits[0][0] in (1, 2), label
    line, column, what = scan("ok\nab​c")[0]
    assert (line, column) == (2, 3) and "ZERO WIDTH SPACE" in what


def test_ordinary_text_with_accents_symbols_and_whitespace_passes():
    assert scan("Birmingham\tNew Street\n£1.39 and café, 15 minutes\r\n- bullet\n") == []


def test_main_exits_one_on_a_hit_and_zero_otherwise(tmp_path, capsys):
    good, bad = tmp_path / "good.md", tmp_path / "bad.md"
    good.write_text("plain\n", encoding="utf-8")
    bad.write_text("pl​ain\n", encoding="utf-8")
    assert main([str(good)]) == 0
    assert main([str(good), str(bad)]) == 1
    assert "bad.md:1:3: invisible character U+200B" in capsys.readouterr().out


def test_every_markdown_file_in_the_repo_is_clean():
    root = Path(__file__).parent.parent
    files = [
        str(p)
        for p in root.rglob("*.md")
        if not any(part in {".venv", "node_modules", ".git", "mlruns"} for part in p.parts)
    ]
    assert files
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_invisible_chars.py"), *files],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout
