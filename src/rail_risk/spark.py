"""Local PySpark session sized for a laptop that is also running other jobs."""

import os
import subprocess
import threading
from pathlib import Path

from pyspark.sql import SparkSession

HOMEBREW_JAVA17 = Path("/opt/homebrew/opt/openjdk@17")


def build_spark(app_name: str = "rail-risk", *, cores: int = 4, driver_memory: str = "6g"):
    """A local[cores] session with a UTC session time zone and adaptive execution on."""
    if "JAVA_HOME" not in os.environ and HOMEBREW_JAVA17.exists():
        os.environ["JAVA_HOME"] = str(HOMEBREW_JAVA17)
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", str(cores * 2))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


class PeakMemory:
    """Samples the resident memory of the Spark driver JVM while a block runs."""

    def __init__(self, spark: SparkSession, interval_s: float = 0.5) -> None:
        self._pid = int(spark.sparkContext._jvm.java.lang.ProcessHandle.current().pid())
        self._interval = interval_s
        self._stop = threading.Event()
        self.peak_mb = 0.0
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _sample(self) -> None:
        out = subprocess.run(  # noqa: S603
            ["/bin/ps", "-o", "rss=", "-p", str(self._pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if out.isdigit():
            self.peak_mb = max(self.peak_mb, int(out) / 1024)

    def _poll(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> "PeakMemory":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
