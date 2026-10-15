import pytest


@pytest.fixture(scope="session")
def spark():
    from rail_risk.spark import build_spark

    session = build_spark("rail-risk-tests", cores=2, driver_memory="1g")
    session.conf.set("spark.sql.shuffle.partitions", "4")
    yield session
    session.stop()
