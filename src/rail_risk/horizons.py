"""Minutes-to-arrival buckets for conformal offsets and threshold recalibration.

The models take minutes to arrival as a number. Offsets and recalibration maps need enough rows
each, so they are fitted per bucket of horizon rather than per grid value, and a request at any
number of minutes uses the bucket it falls in.
"""

import numpy as np

HORIZON_GRID = (5, 15, 30, 60, 90, 120)
# bucket 0 is under 20 minutes, 1 is 20 to 74, 2 is 75 and over
DEFAULT_BUCKET_EDGES = (20.0, 75.0)


def bucket_labels(edges: tuple[float, ...] = DEFAULT_BUCKET_EDGES) -> list[str]:
    labels = [f"under {edges[0]:g}"]
    labels += [f"{lo:g} to {hi - 1:g}" for lo, hi in zip(edges, edges[1:], strict=False)]
    labels.append(f"{edges[-1]:g} or more")
    return labels


def horizon_buckets(
    minutes: np.ndarray | list[float], edges: tuple[float, ...] = DEFAULT_BUCKET_EDGES
) -> np.ndarray:
    """Bucket index per value: the number of edges at or below it."""
    return np.searchsorted(np.asarray(edges), np.asarray(minutes, dtype=float), side="right")
