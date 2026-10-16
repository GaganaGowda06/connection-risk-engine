"""Streamlit page: arrival range and delay chances for a live service.

Run with `rail-risk ui`. The page talks to the API over HTTP and never loads a model.
"""

import streamlit as st

from rail_risk.ui import client
from rail_risk.ui.badges import badges

HOW_IT_WORKS = """
**Where the numbers come from.** Live train movements come from National Rail's Darwin feed,
which reports where each train is and how late it is. The model also learns from the Historical
Service Performance record of past journeys on the same route and time of day.

**What the range means.** The range is built so that about 8 in 10 trains arrive inside it. That
also means about 2 in 10 do not: some arrive earlier and some later. The chances of being 15, 30
or 60 or more minutes late are the model's estimates for this train, from where it is now.

**Delay Repay.** The amount shown is an expected value, the average payout over what could
happen. It is not a promise and it is not what you will be paid on any one day. It uses the
operator's own published tiers when we could verify them, and the fare you type in.

**Known limits.**
- Only two routes have been checked so far: Birmingham New Street to London Euston, and
  Birmingham Aston to Birmingham New Street. This page lists the first.
- The model learned from summer and early autumn. Winter and storm weather are not covered yet.
- The model was trained at forecast points from 5 to 120 minutes before arrival, and is given
  the real minutes to go. Closer than 5 minutes it has not been tested.
- The live feed can lag. When it is more than 10 minutes old, the page says so.
- Connections between trains are not included yet.
"""


def _money(value: float) -> str:
    return f"£{value:,.2f}"


def _show_risk(body: dict) -> None:
    status = body.get("status")
    if status == "cancelled":
        reason = body.get("cancellation_reason")
        st.error(
            "This train is cancelled." + (f" Darwin gives reason code {reason}." if reason else "")
        )
        return
    if status == "arrived":
        delay = body.get("median_delay_minutes")
        st.success(
            f"This train has arrived at {body.get('median_arrival')}, "
            f"{delay:+.0f} minutes against its timetable."
        )
        return
    st.markdown(f"# Expected {body['arrival_range_80']}")
    st.caption(
        f"Scheduled arrival {body['scheduled_arrival']}. Most likely arrival "
        f"{body['median_arrival']}, about {body['median_delay_minutes']:+.0f} minutes against "
        f"the timetable. About 8 in 10 trains land inside this range."
    )
    st.subheader("Chance of arriving late")
    for minutes in (15, 30, 60):
        p = body[f"p_delay_at_least_{minutes}"]
        st.progress(min(max(p, 0.0), 1.0), text=f"{minutes} or more minutes late: {p:.0%}")


def _show_repay(body: dict, fare: float | None) -> None:
    repay = body.get("delay_repay") or {}
    st.subheader("Delay Repay")
    if repay.get("amount_gbp") is not None:
        status = repay.get("toc_status", "")
        st.metric(
            "Expected Delay Repay on this ticket",
            _money(repay["amount_gbp"]),
            help=f"Operator rules {status}. An average over what could happen, not a promise.",
        )
        st.caption(f"Operator {repay.get('toc')}: rules {status}.")
    elif fare is None:
        st.caption("Enter your ticket price to see the expected Delay Repay.")
    else:
        st.caption(f"No amount shown: {repay.get('reason', 'rules not available')}.")
    if repay.get("caveat"):
        st.caption(repay["caveat"])


def main() -> None:
    st.set_page_config(page_title="Train arrival range", page_icon="🚆", layout="centered")
    st.title("When will my train arrive?")

    try:
        routes = client.get_routes()
    except client.ApiDown:
        st.error("The risk service is not reachable right now. Please try again in a minute.")
        st.stop()
        return

    usable = [r for r in routes if r["model_trained_on_route"]] or routes
    labels = {r["id"]: f"{r['origin']} to {r['destination']}" for r in usable}
    route_id = st.selectbox("Route", list(labels), format_func=lambda k: labels[k])

    try:
        services = client.get_services(route_id)
    except client.ApiDown:
        st.error("Could not load the list of trains. Please try again in a minute.")
        st.stop()
        return
    if not services:
        st.info("No trains are due on this route in the next 3 hours.")
        return

    def describe(rid: str) -> str:
        s = next(s for s in services if s["rid"] == rid)
        tag = " (cancelled)" if s["cancelled"] else ""
        return (
            f"Arrives {s['scheduled_arrival']}, {s['toc'] or 'operator unknown'}, "
            f"from {s['origin']} at {s['scheduled_departure']}{tag}"
        )

    rid = st.selectbox("Train", [s["rid"] for s in services], format_func=describe)
    col_fare, col_ticket = st.columns(2)
    fare_value = col_fare.number_input(
        "Ticket price in pounds (optional)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0
    )
    ticket = col_ticket.selectbox("Ticket type", ["single", "return"])
    fare = fare_value if fare_value > 0 else None

    try:
        status, body = client.risk_with_retry(rid, fare, ticket)
    except client.ApiDown:
        st.error("The risk service stopped answering. Please try again in a minute.")
        return

    if status == 202:
        st.info("We are still gathering live data for this train. Try again in a moment.")
        st.button("Try again")
        return
    if status == 404:
        st.warning("We have no live information for this train right now.")
        return
    if status == 504:
        st.warning("That took too long. Please try again.")
        return
    if status != 200:
        st.error(f"Something went wrong (code {status}).")
        return

    _show_risk(body)
    if body.get("status") in ("running", "not_started", "arrived"):
        _show_repay(body, fare)
    for badge in badges(body):
        message = f"**{badge.title}.** {badge.text}"
        (st.warning if badge.level == "warning" else st.info)(message)
    with st.expander("How this works"):
        st.markdown(HOW_IT_WORKS)
    st.caption(
        f"Model run {str(body.get('model_run_id'))[:8]}. "
        f"Live data age {body['data_quality']['live_data_age_minutes']} minutes."
    )


main()
