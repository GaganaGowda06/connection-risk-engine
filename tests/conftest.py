import pytest


@pytest.fixture(scope="session")
def spark():
    from rail_risk.spark import build_spark

    session = build_spark("rail-risk-tests", cores=2, driver_memory="1g")
    session.conf.set("spark.sql.shuffle.partitions", "4")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def bundle_pair():
    from world import build_bundle

    return build_bundle()


@pytest.fixture(scope="module")
def features(spark):
    from rail_risk.live import FeatureService
    from world import history_labels

    route_dest = {"bhm-eus": "EUS", "ast-bhm": "BHM", "bhm-man": "MAN"}
    return FeatureService.from_labels(spark, route_dest, history_labels(spark))
