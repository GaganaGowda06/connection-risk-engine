"""The hours removed in the validation replay used to fix the guard thresholds."""

import pandas as pd

GAPS = [
    (pd.Timestamp("2026-10-02 09:00", tz="UTC"), pd.Timestamp("2026-10-02 10:00", tz="UTC")),
    (pd.Timestamp("2026-10-03 15:00", tz="UTC"), pd.Timestamp("2026-10-03 17:00", tz="UTC")),
]
