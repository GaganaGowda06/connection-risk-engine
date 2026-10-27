import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_commit_msg", Path(__file__).parent.parent / "scripts" / "check_commit_msg.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_rejects_trailers_and_credits(tmp_path):
    for bad in (
        "x\n\n" + "Co-Auth" + "ored-By: Someone <a@b.c>",
        "Generated" + " with a tool",
        "see cl" + "aude.ai",
        "ANTHR" + "OPIC",
    ):
        p = tmp_path / "m"
        p.write_text(bad)
        assert mod.main(str(p)) == 1


def test_accepts_a_plain_message(tmp_path):
    p = tmp_path / "m"
    p.write_text("Stage 5 rescored after the label fix\n")
    assert mod.main(str(p)) == 0
