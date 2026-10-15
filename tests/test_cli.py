import json
from pathlib import Path

import pytest

from rail_risk.cli import main
from rail_risk.logging import configure_logging

ROUTES = Path(__file__).parent.parent / "config" / "routes.yaml"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    for k in (
        "RAIL_RDM_DARWIN_USER",
        "RAIL_RDM_DARWIN_PASSWORD",
        "RAIL_RDM_DARWIN_GROUP",
        "RAIL_RDM_HSP_KEY",
        "RAIL_LOG_JSON",
        "RAIL_LOG_LEVEL",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RAIL_ROUTES_FILE", str(ROUTES))


def test_check_ok_hides_secret(monkeypatch, capsys):
    monkeypatch.setenv("RAIL_RDM_DARWIN_PASSWORD", "s3cret-value")
    monkeypatch.setenv("RAIL_RDM_DARWIN_USER", "user-value")
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "config_ok" in out
    assert "s3cret-value" not in out
    assert "user-value" not in out
    assert "darwin_credentials_set=True" in out
    assert "bhm-eus" in out


def test_check_json_output(monkeypatch, capsys):
    monkeypatch.setenv("RAIL_LOG_JSON", "true")
    assert main(["check"]) == 0
    rec = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rec["event"] == "config_ok"
    assert rec["darwin_credentials_set"] is False
    assert rec["timestamp"].endswith("Z")


def test_check_missing_routes_file(monkeypatch, capsys):
    monkeypatch.setenv("RAIL_ROUTES_FILE", "nope.yaml")
    monkeypatch.setenv("RAIL_RDM_HSP_KEY", "s3cret-value")
    assert main(["check"]) == 1
    out = capsys.readouterr().out
    assert "config_invalid" in out
    assert "s3cret-value" not in out


def test_check_bad_log_level(monkeypatch, capsys):
    monkeypatch.setenv("RAIL_LOG_LEVEL", "LOUD")
    assert main(["check"]) == 1
    assert "config_invalid" in capsys.readouterr().out


def test_configure_logging_rejects_unknown_level():
    with pytest.raises(ValueError):
        configure_logging("LOUD")


def test_ingest_hsp_passes_newest_first_and_exit_codes(monkeypatch):
    import rail_risk.ingest.hsp as hsp

    monkeypatch.setenv("RAIL_RDM_HSP_KEY", "k")
    monkeypatch.setenv("RAIL_DATA_DIR", "data")
    seen = {}

    def fake_backfill(route, start, end, hours, client, manifest, **kw):
        seen.update(kw, route=route.id, start=start, end=end, hours=hours)
        return hsp.Counters(failed_windows=seen.get("fail", 0))

    monkeypatch.setattr(hsp, "backfill", fake_backfill)
    monkeypatch.setattr(hsp, "Manifest", lambda path: type("M", (), {"close": lambda s: None})())
    argv = ["ingest-hsp", "--route", "bhm-eus", "--start", "2026-09-01", "--end", "2026-09-30"]
    assert main([*argv, "--newest-first", "--hours", "07-09"]) == 0
    assert seen["newest_first"] is True
    assert list(seen["hours"]) == [7, 8]
    assert main(argv) == 0
    assert seen["newest_first"] is False
    assert (
        main(["ingest-hsp", "--route", "nope", "--start", "2026-09-01", "--end", "2026-09-02"]) == 1
    )


@pytest.mark.parametrize(
    ("min_days", "dirty", "expect_report"),
    [(60, False, False), (3, False, True), (3, True, False)],
)
def test_train_guard_blocks_the_report_below_the_threshold_or_on_a_dirty_tree(
    monkeypatch, tmp_path, capsys, min_days, dirty, expect_report
):
    from datetime import date, timedelta

    import pandas as pd

    import rail_risk.evaluation as ev
    import rail_risk.models.pipeline as pipe

    days = [date(2026, 9, 1) + timedelta(days=i) for i in range(30)]
    snaps = pd.DataFrame({"route_id": "r1", "service_date": days})
    monkeypatch.setattr(ev, "load_snapshots", lambda path: snaps)
    monkeypatch.setattr(ev, "complete_days", lambda *a, **k: days)
    seen = {}
    monkeypatch.setattr(pipe, "run_training", lambda *a, **k: ({"train_days": 4}, object()))

    def fake_log(result, fitted, **kw):
        seen["tag"] = kw["tag"]
        seen["sha"], seen["dirty"] = kw["git_sha"], kw["git_dirty"]
        return "run123"

    monkeypatch.setattr(pipe, "log_to_mlflow", fake_log)
    monkeypatch.setattr(pipe, "git_state", lambda: ("a" * 40, dirty))
    written = []
    monkeypatch.setattr(
        pipe, "write_model_report", lambda r, path, run_id, sha="": written.append(path)
    )
    cfg = tmp_path / "train.yaml"
    cfg.write_text(f"min_train_days: {min_days}\nseed: 1\n")
    root = Path(__file__).parent.parent / "config"
    code = main(
        [
            "train",
            "--config", str(cfg),
            "--split-file", str(root / "split.yaml"),
            "--bands-file", str(root / "delay_bands.yaml"),
            "--report", str(tmp_path / "model_results.md"),
        ]
    )  # fmt: skip
    assert code == 0
    assert seen["tag"] == ("reportable" if expect_report else "smoke")
    assert seen["sha"] == "a" * 40 and seen["dirty"] is dirty
    assert bool(written) is expect_report
    assert "run123" in capsys.readouterr().out
