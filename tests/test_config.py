from pathlib import Path

import pytest
from pydantic import ValidationError

from rail_risk.config import Route, RouteSet, Settings, load_routes


def route(**kw):
    base = {"id": "r1", "origin": "BHM", "destination": "EUS"}
    return Route(**{**base, **kw})


@pytest.mark.parametrize("bad", ["bhm", "BH", "BHMX", "B1M", "", "BH ", "ÉUS"])
@pytest.mark.parametrize("field", ["origin", "destination", "via_hub"])
def test_bad_crs_rejected(field, bad):
    with pytest.raises(ValidationError):
        route(**{field: bad})


@pytest.mark.parametrize(
    "kw",
    [
        {"destination": "BHM"},
        {"via_hub": "BHM"},
        {"via_hub": "EUS"},
    ],
)
def test_repeated_station_rejected(kw):
    with pytest.raises(ValidationError, match="repeats station"):
        route(**kw)


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate route ids"):
        RouteSet(routes=[route(), route(destination="MAN")])


def test_empty_routes_rejected():
    with pytest.raises(ValidationError, match="must not be empty"):
        RouteSet(routes=[])


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_routes(tmp_path / "nope.yaml")


@pytest.mark.parametrize("text", ["- a\n- b\n", "just a string", "", "a: [unclosed"])
def test_non_mapping_yaml(tmp_path, text):
    p = tmp_path / "r.yaml"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_routes(p)


ROUTES_FILE = Path(__file__).parent.parent / "config" / "routes.yaml"


def test_shipped_table_has_25_routes_11_operators_and_the_held_out_set():
    routes = load_routes(ROUTES_FILE)
    table = [r for r in routes.routes if not r.legacy]
    assert len(table) == 25
    assert len({r.operator for r in table}) == 11
    assert {r.hub for r in table} == {"London", "Birmingham", "Manchester", "Glasgow", "Bristol"}
    assert {r.length for r in table} == {"short", "medium", "long"}
    assert {r.id for r in table if r.held_out} == {
        "lst-cbg", "myb-bmo", "bhm-liv", "man-shf", "glq-abd", "bri-cdf", "bri-wsm",
    }  # fmt: skip
    assert all(r.operator and r.toc and r.length and r.hub for r in table)


def test_legacy_routes_are_flagged_and_outside_the_table():
    routes = load_routes(ROUTES_FILE).routes
    assert {r.id for r in routes if r.legacy} == {"ast-eus-via-bhm", "bhm-lds", "bhm-glc"}


def test_interchanges_point_at_table_routes():
    routes = load_routes(ROUTES_FILE)
    ids = {r.id for r in routes.routes if not r.legacy}
    assert len(routes.interchanges) == 4
    assert {i.outbound for i in routes.interchanges} <= ids


def test_secrets_masked_in_repr():
    s = Settings(
        _env_file=None,
        rdm_darwin_password="topsecret",  # noqa: S106
        rdm_darwin_user="someuser",
        rdm_hsp_key="hspsecret",
    )
    assert "topsecret" not in repr(s)
    assert "hspsecret" not in str(s)
    assert "someuser" not in repr(s)
    assert s.rdm_darwin_password.get_secret_value() == "topsecret"


def test_unknown_log_level_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="LOUD")


def test_empty_env_value_is_unset(monkeypatch):
    monkeypatch.setenv("RAIL_RDM_DARWIN_PASSWORD", "")
    assert Settings(_env_file=None).rdm_darwin_password is None
