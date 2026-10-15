"""MLflow model registry on the sqlite store: register a run, move the alias, load by alias."""

import os
from dataclasses import dataclass
from pathlib import Path

from rail_risk.models.bundle import ModelBundle, load_baseline, load_bundle

MODEL_NAME = "connection-risk"
ALIAS = "champion"


def _client(tracking_dir: Path):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow import MlflowClient

    uri = f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return mlflow, MlflowClient(tracking_uri=uri, registry_uri=uri)


def register_champion(
    tracking_dir: Path, run_id: str, name: str = MODEL_NAME, alias: str = ALIAS
) -> str:
    """Register the run's `model` artefact as a new version and point the alias at it."""
    _, client = _client(tracking_dir)
    run = client.get_run(run_id)
    if run.data.tags.get("stage") != "reportable":
        raise ValueError(f"run {run_id} is tagged {run.data.tags.get('stage')!r}, not reportable")
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(name)
    except MlflowException:  # already registered
        pass
    # the run's plain `model` artefact folder is the source; it is not an MLflow logged model
    version = client.create_model_version(
        name, source=f"{run.info.artifact_uri}/model", run_id=run_id, tags={"run_id": run_id}
    )
    client.set_registered_model_alias(name, alias, version.version)
    return str(version.version)


def _local_source(source: str, tracking_dir: Path) -> str:
    """The artefact folder for a version, even when the store has moved.

    The registry records the absolute path of the machine that trained the model. When that
    path does not exist (the store is mounted elsewhere, as in a container) the part after
    `/artifacts/` is looked up under this tracking directory instead.
    """
    from urllib.parse import unquote, urlparse

    parsed = urlparse(source)
    if parsed.scheme not in ("", "file"):
        return source
    path = Path(unquote(parsed.path))
    if path.exists() or "/artifacts/" not in str(path):
        return source
    relative = str(path).split("/artifacts/", 1)[1]
    candidate = tracking_dir.resolve() / "artifacts" / relative
    return str(candidate) if candidate.exists() else source


@dataclass
class Loaded:
    bundle: ModelBundle
    baseline: object | None
    run_id: str
    version: str
    alias: str
    tags: dict[str, str]


def load_alias(tracking_dir: Path, name: str = MODEL_NAME, alias: str = ALIAS) -> Loaded:
    """Resolve the alias, download the version's artefacts and load the bundle."""
    mlflow, client = _client(tracking_dir)
    version = client.get_model_version_by_alias(name, alias)
    folder = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=_local_source(version.source, tracking_dir)
        )
    )
    run = client.get_run(version.run_id)
    return Loaded(
        bundle=load_bundle(folder),
        baseline=load_baseline(folder),
        run_id=version.run_id,
        version=str(version.version),
        alias=alias,
        tags=dict(run.data.tags),
    )
