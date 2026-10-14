import json
from pathlib import Path

import pytest

from rail_risk.cli import main
from rail_risk.logging import configure_logging

ROUTES = Path(__file__).parent.parent / "config" / "routes.yaml"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    for k in ("RAIL_RDM_DARWIN_KEY", "RAIL_RDM_HSP_KEY", "RAIL_LOG_JSON", "RAIL_LOG_LEVEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RAIL_ROUTES_FILE", str(ROUTES))


def test_check_ok_hides_secret(monkeypatch, capsys):
    monkeypatch.setenv("RAIL_RDM_DARWIN_KEY", "s3cret-value")
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "config_ok" in out
    assert "s3cret-value" not in out
    assert "bhm-eus" in out


def test_check_json_output(monkeypatch, capsys):
    monkeypatch.setenv("RAIL_LOG_JSON", "true")
    assert main(["check"]) == 0
    rec = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rec["event"] == "config_ok"
    assert rec["darwin_key_set"] is False
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
