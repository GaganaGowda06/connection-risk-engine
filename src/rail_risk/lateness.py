"""Observed-lateness groups for Mondrian conformal calibration and group coverage reporting.

The group comes only from `last_stop_lateness_min`, which is known at the snapshot time T. It
never uses the label or the label band.
"""

import numpy as np
import pandas as pd

DEFAULT_EDGES = (5.0, 15.0)


def group_labels(edges: tuple[float, ...] = DEFAULT_EDGES) -> list[str]:
    """['not_started', '0-4', '5-14', '15+'] for the default edges. Early running joins '0-4'."""
    labels = ["not_started"]
    lower = None
    for edge in edges:
        labels.append(f"0-{int(edge) - 1}" if lower is None else f"{int(lower)}-{int(edge) - 1}")
        lower = edge
    labels.append(f"{int(edges[-1])}+")
    return labels


def lateness_groups(df: pd.DataFrame, edges: tuple[float, ...] = DEFAULT_EDGES) -> np.ndarray:
    """Group index per row: 0 not started (no lateness known), then one per edge interval."""
    if "last_stop_lateness_min" not in df.columns:
        return np.zeros(len(df), dtype=int)
    late = df["last_stop_lateness_min"].to_numpy(dtype=float)
    groups = 1 + np.searchsorted(np.asarray(edges), np.nan_to_num(late), side="right")
    return np.where(np.isnan(late), 0, groups).astype(int)


def lateness_base(df: pd.DataFrame) -> np.ndarray:
    """Lateness carried to the destination: last known lateness, 0 if the train has not started."""
    if "last_stop_lateness_min" not in df.columns:
        return np.zeros(len(df))
    return df["last_stop_lateness_min"].fillna(0.0).to_numpy(dtype=float)


def merge_groups(
    counts: dict[int, int], min_rows: int
) -> tuple[dict[int, int], list[tuple[list[int], list[int]]]]:
    """Merge groups below `min_rows` into an adjacent group until none is below or one remains.

    Groups are ordered by lateness, so only neighbours merge. A small group joins the neighbour
    with fewer rows, ties going to the lower group. Returns (group -> representative group, list
    of (small members, receiving members) merges in order).
    """
    blocks = [([k], counts[k]) for k in sorted(counts)]
    merges: list[tuple[list[int], list[int]]] = []
    while len(blocks) > 1:
        small = [i for i, (_, n) in enumerate(blocks) if n < min_rows]
        if not small:
            break
        i = min(small, key=lambda j: (blocks[j][1], j))
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
        j = min(neighbours, key=lambda k: (blocks[k][1], k))
        merges.append((list(blocks[i][0]), list(blocks[j][0])))
        lo, hi = sorted((i, j))
        blocks[lo] = (blocks[lo][0] + blocks[hi][0], blocks[lo][1] + blocks[hi][1])
        del blocks[hi]
    return {m: min(members) for members, _ in blocks for m in members}, merges
