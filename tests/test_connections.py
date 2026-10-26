"""Stage 5: labels, identity-free features, MCT fail-closed, online recalibration, baseline."""

import numpy as np
import pandas as pd
import pytest

from rail_risk.connections import baseline as BL
from rail_risk.connections import build as B
from rail_risk.connections import models as M
from rail_risk.connections import online as O


def ts(s):
    return pd.Timestamp(s, tz="UTC")


NAT = pd.Series([pd.NaT], dtype="datetime64[us, UTC]")


def frame(**kw):
    base = {
        "act_arr_in": [ts("2026-10-01 10:00")], "act_dep_out": [ts("2026-10-01 10:20")],
        "in_cancel_final": [False], "in_ran": [True], "out_cancel_final": [False],
        "out_ran": [True], "mct_min": [10.0], "plan_arr_in": [ts("2026-10-01 10:00")],
        "plan_dep_out": [ts("2026-10-01 10:20")], "gap": [20.0],
    }  # fmt: skip
    base.update(kw)
    return pd.DataFrame(base)


def test_miss_when_arrival_plus_mct_is_after_departure():
    df, _ = B.label(frame(act_arr_in=[ts("2026-10-01 10:15")]), {})
    assert df["miss"].iloc[0] == 1.0
    df, _ = B.label(frame(act_arr_in=[ts("2026-10-01 10:10")]), {})
    assert df["miss"].iloc[0] == 0.0  # arrival plus MCT equal to departure is a connection


def test_late_outbound_saves_the_connection():
    df, _ = B.label(
        frame(act_arr_in=[ts("2026-10-01 10:15")], act_dep_out=[ts("2026-10-01 10:28")]), {}
    )
    assert df["miss"].iloc[0] == 0.0


def test_cancelled_inbound_is_a_miss_cancelled_outbound_has_no_label():
    df, _ = B.label(frame(act_arr_in=NAT, in_cancel_final=[True], in_ran=[False]), {})
    assert df["miss"].iloc[0] == 1.0
    df, _ = B.label(frame(act_dep_out=NAT, out_cancel_final=[True]), {})
    assert df["out_cancel"].iloc[0] == 1 and np.isnan(df["miss"].iloc[0])


def test_service_with_no_actual_and_no_cancellation_is_dropped():
    st: dict = {}
    df, st = B.label(frame(act_arr_in=NAT, in_ran=[False]), st)
    assert df.empty and st["unknown_inbound_dropped"] == 1


def test_unknown_mct_is_nan_never_a_default():
    table = pd.DataFrame({"crs": ["BHM"], "arr_toc": ["*"], "dep_toc": ["*"], "minutes": [12.0]})
    pairs = pd.DataFrame({"hub": ["BHM", "XXX"], "in_toc": ["XC", "XC"], "out_toc": ["VT", "VT"]})
    out = B.pair_mct(pairs, table)
    assert out.iloc[0] == 12.0 and np.isnan(out.iloc[1])


def test_feature_names_carry_no_identity():
    for names in (M.FEATS_A, M.FEATS_D, M.FEATS_B, M.FEATS_C):
        M.check_features(names)
    with pytest.raises(ValueError):
        M.check_features(["hub_size"])
    with pytest.raises(ValueError):
        M.check_features(["in_toc"])


def test_held_out_groups_are_stable_and_include_held_routes():
    held = {"LST|CBG"}
    assert B.group_is_held("LST|CBG", 6, 0, held)
    a = [B.group_is_held(f"A{i}|B{i}", 6, 0, held) for i in range(600)]
    assert a == [B.group_is_held(f"A{i}|B{i}", 6, 0, held) for i in range(600)]
    assert 0.08 < np.mean(a) < 0.26


def test_gap_bucket_uses_pooled_rate_for_an_unseen_hub():
    tr = pd.DataFrame({"hub": ["A"] * 40, "slack_min": [3.0] * 40, "miss": [1.0] * 20 + [0.0] * 20})
    m = BL.fit(tr)
    test = pd.DataFrame({"hub": ["B"], "slack_min": [3.0]})
    assert abs(BL.predict(m, test)[0] - 21 / 42) < 1e-9


def _stream(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 5 * 86400, n))
    hub = rng.choice(["A", "B"], n)
    p = np.full(n, 0.1)
    y = (rng.uniform(size=n) < 0.3).astype(float)
    return hub, t, p, y, t + rng.uniform(300, 1800, n)


def test_online_zero_step_is_the_global_model():
    hub, t, p, y, r = _stream()
    q, _ = O.run(hub, t, p, y, r, 0.0)
    assert (q == p).all()


def test_online_uses_only_departed_outcomes_and_moves_toward_the_rate():
    hub, t, p, y, r = _stream()
    q, used = O.run(hub, t, p, y, r, 0.01)
    assert np.all(~np.isfinite(used) | (used < t))
    assert q[-200:].mean() > 0.15  # the miss rate is 0.3, the global model said 0.1


def test_online_future_labels_cannot_change_earlier_rows():
    hub, t, p, y, r = _stream()
    cut = 2 * 86400
    y2 = y.copy()
    y2[r >= cut] = 1 - y2[r >= cut]
    q0, _ = O.run(hub, t, p, y, r, 0.01)
    q1, _ = O.run(hub, t, p, y2, r, 0.01)
    assert (q0[t < cut] == q1[t < cut]).all() and (q0[t > cut + 20000] != q1[t > cut + 20000]).any()


def test_hubs_do_not_share_state():
    hub, t, p, y, r = _stream()
    y2 = y.copy()
    y2[hub == "B"] = 1.0 - y2[hub == "B"]
    q0, _ = O.run(hub, t, p, y, r, 0.01)
    q1, _ = O.run(hub, t, p, y2, r, 0.01)
    assert (q0[hub == "A"] == q1[hub == "A"]).all()


def test_unlabelled_rows_never_update():
    hub, t, p, y, r = _stream()
    y2 = y.copy()
    y2[::2] = np.nan
    q, used = O.run(hub, t, p, y2, r, 0.01)
    assert np.isfinite(q).all()


def test_ran_inbound_with_no_arrival_and_no_cancellation_is_unknown_not_a_miss():
    st: dict = {}
    df, st = B.label(frame(act_arr_in=NAT, in_ran=[True], in_cancel_final=[False]), st)
    assert df.empty and st["unknown_inbound_dropped"] == 1
    df, _ = B.label(frame(act_arr_in=NAT, in_ran=[True], in_cancel_final=[True]), {})
    assert df["miss"].iloc[0] == 1.0


def test_stored_files_are_relabelled_on_read():
    from rail_risk.connections import data as DD

    df = pd.DataFrame(
        {
            "act_arr_in": [NAT[0], NAT[0], ts("2026-10-01 10:00")],
            "in_cancel_final": [False, True, False],
            "miss": [1.0, 1.0, 0.0],
        }
    )
    assert DD.apply_label_rule(df)["miss"].tolist() == [1.0, 0.0]
