import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rail_risk.logging import LEVELS

CRS_PATTERN = re.compile(r"^[A-Z]{3}$")


class Route(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    origin: str
    destination: str
    via_hub: str | None = None

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("route id must not be blank")
        return v

    @field_validator("origin", "destination", "via_hub")
    @classmethod
    def _crs_code(cls, v: str | None) -> str | None:
        if v is not None and not CRS_PATTERN.fullmatch(v):
            raise ValueError(f"CRS code must be three uppercase letters, got {v!r}")
        return v

    @model_validator(mode="after")
    def _no_repeated_station(self) -> "Route":
        stations = [self.origin, self.destination]
        if self.via_hub is not None:
            stations.append(self.via_hub)
        repeated = sorted({s for s in stations if stations.count(s) > 1})
        if repeated:
            raise ValueError(f"route {self.id!r} repeats station {', '.join(repeated)}")
        return self


class RouteSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    routes: list[Route]

    @field_validator("routes")
    @classmethod
    def _non_empty_unique(cls, v: list[Route]) -> list[Route]:
        if not v:
            raise ValueError("routes must not be empty")
        ids = [r.id for r in v]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate route ids: {', '.join(dupes)}")
        return v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAIL_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    rdm_darwin_user: SecretStr | None = None
    rdm_darwin_password: SecretStr | None = None
    rdm_darwin_group: str | None = None
    rdm_hsp_key: SecretStr | None = None
    routes_file: Path = Path("config/routes.yaml")
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, v: str) -> str:
        name = v.upper()
        if name not in LEVELS:
            raise ValueError(f"unknown log level {v!r}; expected one of {', '.join(LEVELS)}")
        return name


def load_routes(path: Path | str) -> RouteSet:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"routes file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"routes file {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"routes file {path} must be a YAML mapping with a 'routes' key")
    return RouteSet.model_validate(data)
