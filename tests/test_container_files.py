import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def test_dockerignore_keeps_secrets_data_and_state_out_of_the_build_context():
    lines = {
        line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines() if line.strip()
    }
    assert {".env", "data/", "mlruns/", ".git/", ".venv/"} <= lines


def test_dockerfile_runs_as_non_root_with_a_healthcheck_and_no_secrets():
    text = (ROOT / "Dockerfile").read_text()
    assert "USER app" in text and "HEALTHCHECK" in text
    assert "--locked" in text  # the install comes from uv.lock
    assert not re.search(r"(?im)^\s*(ENV|ARG)\s+\S*(PASSWORD|SECRET|TOKEN|API_?KEY)", text)
    assert "COPY .env" not in text and "COPY . " not in text


def test_compose_mounts_state_read_only_and_takes_credentials_from_env_file_only():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    api, ui, capture = (compose["services"][k] for k in ("api", "ui", "capture"))
    assert "./data:/app/data:ro" in api["volumes"] and "./mlruns:/app/mlruns:ro" in api["volumes"]
    assert all(p.startswith("127.0.0.1:") for p in api["ports"] + ui["ports"])
    assert ui["environment"]["RAIL_API_URL"] == "http://api:8000"
    assert capture["profiles"] == ["capture"] and capture["env_file"][0]["path"] == ".env"
    for name in ("api", "ui"):
        assert "env_file" not in compose["services"][name]
    flat = (ROOT / "docker-compose.yml").read_text().lower()
    assert "password" not in flat and "apikey" not in flat
