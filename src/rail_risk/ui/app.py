"""Streamlit page: arrival range and delay chances for a live service.

Run with `rail-risk ui`. The page talks to the API over HTTP and never loads a model.
"""

import re

import streamlit as st

from rail_risk.ui import client
from rail_risk.ui.badges import badges
from rail_risk.ui.words import miss_sentence

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


def arrival_tab(routes: list[dict]) -> None:
    usable = [r for r in routes if r["model_trained_on_route"]] or routes
    labels = {r["id"]: f"{r['origin']} to {r['destination']}" for r in usable}
    route_id = st.selectbox("Route", list(labels), format_func=lambda k: labels[k])

    try:
        services = client.get_services(route_id)
    except client.ApiDown:
        st.error("Could not load the list of trains. Please try again in a minute.")
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


CONNECTION_HOW = """
**What this shows.** You arrive at Birmingham New Street from Aston and change for a train to
London Euston. We forecast when your first train gets in, then simulate many ways the day could go
to see how often you would not make the connection.

**Assumptions.**
- The change at New Street needs 10 minutes. That figure is our assumption, not a published one.
- The first train and the Euston train are treated as independent. If something goes wrong for
  both at once, such as a signal failure, this page does not know.
- Delays on the Euston train come from past trains at the same hour, not from today's running.
- If you miss it, you take the next train you can catch. Some days none is left, and the page
  says so.
- The alert level comes from an assumed cost of 1 for a needless warning and 5 for a missed one.
"""

HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _show_connection(body: dict, fare: float | None) -> None:
    status = body.get("status")
    if status == "cancelled":
        st.error(body.get("refusal") or "The train from Aston is cancelled.")
        return
    if status == "refused":
        st.warning(body.get("refusal", "That connection cannot be made."))
        first = body.get("first_feasible_departure")
        if first:
            st.info(f"The first Euston train you could make leaves New Street at {first}.")
        return
    out, inbound = body["outbound"], body["inbound"]
    st.markdown(f"## {miss_sentence(body['p_miss'], out['scheduled_departure'])}")
    if body.get("alert"):
        st.error(
            f"**Your connection is at risk.** The chance of missing it, {body['p_miss']:.0%}, is "
            f"at or above our alert level of {body['alert_threshold']:.0%}. Check the board for "
            "a later train."
        )
    where = {
        "arrived": "Your train from Aston has arrived at New Street.",
        "approaching": "Your train from Aston is due at New Street at "
        f"{inbound['scheduled_arrival']}.",
        "due_or_late": f"Your train from Aston was due at {inbound['scheduled_arrival']} and "
        "is about to arrive or running late.",
    }[inbound["state"]]
    st.caption(
        f"{where} Change for the {out['scheduled_departure']} to Euston "
        f"({out['scheduled_gap_minutes']:.0f} minute gap on the timetable)"
        + (", the first one you can make." if out["chosen_automatically"] else ".")
    )
    st.subheader("Arrival at Euston")
    if body.get("arrival_range_80"):
        st.markdown(f"# Expected {body['arrival_range_80']}")
        st.caption(
            f"Scheduled {out['scheduled_arrival']}. Most likely {body['median_arrival']}. "
            "About 8 in 10 journeys land inside this range."
        )
    else:
        st.warning("We cannot give an arrival time: in every way we simulated, no train was left.")
    stranded = body.get("p_stranded") or 0.0
    if stranded >= 0.005:
        st.caption(
            f"If you miss it, there is about a {stranded:.0%} chance that no later train is "
            "catchable that day."
        )
    repay = body.get("delay_repay") or {}
    st.subheader("Delay Repay")
    if repay.get("amount_gbp") is not None:
        st.metric(
            "Expected Delay Repay on this journey",
            _money(repay["amount_gbp"]),
            help="An average over what could happen. It is information only, not a promise.",
        )
        st.caption(repay["stranded_assumption"])
    elif fare is None:
        st.caption("Enter your ticket price to see the expected Delay Repay.")
    else:
        st.caption(f"No amount shown: {repay.get('reason', 'rules not available')}.")
    for badge in badges(body):
        (st.warning if badge.level == "warning" else st.info)(f"**{badge.title}.** {badge.text}")
    with st.expander("Connection time sensitivity and assumptions"):
        ct = body["connection_time"]
        st.markdown(
            f"The change at {ct['station']} is set to {ct['minutes']} minutes, which is "
            f"**{ct['status']}**. How the answer moves with that figure:"
        )
        rows = []
        for r in body["sensitivity"]:
            payout = r["expected_payout_gbp"]
            rows.append(
                {
                    "Change time (minutes)": r["minutes"],
                    "Chance you miss it": f"{r['p_miss']:.0%}" if r["feasible"] else "not possible",
                    "Arrival at Euston": r["arrival_range_80"] or "n/a",
                    "Expected Delay Repay": _money(payout) if payout is not None else "n/a",
                }
            )
        st.table(rows)
        st.markdown("\n".join(f"- {a}" for a in body["assumptions"]))
    with st.expander("How this works"):
        st.markdown(CONNECTION_HOW)
    st.caption(
        f"Model run {str(body.get('model_run_id'))[:8]}. "
        f"Live data age {body['data_quality']['live_data_age_minutes']} minutes."
    )


def connection_tab() -> None:
    st.subheader("Aston to Euston, changing at Birmingham New Street")
    try:
        inbound = client.get_inbound()
    except client.ApiDown:
        st.error("Could not load the list of trains. Please try again in a minute.")
        return
    if not inbound:
        st.info("No trains from Aston are due at New Street in the next 2 hours.")
        return

    def describe(rid: str) -> str:
        s = next(s for s in inbound if s["rid"] == rid)
        tag = " (cancelled)" if s["cancelled"] else ""
        return (
            f"Arrives New Street {s['scheduled_arrival']}, {s['toc'] or 'operator unknown'}, "
            f"from Aston at {s['scheduled_departure']}{tag}"
        )

    rid = st.selectbox("Your train from Aston", [s["rid"] for s in inbound], format_func=describe)
    wanted = st.text_input(
        "Euston train leaving New Street at (HH:MM, optional)",
        value="",
        help="Leave empty to use the first Euston train you can make.",
    ).strip()
    col_fare, col_ticket = st.columns(2)
    fare_value = col_fare.number_input(
        "Ticket price in pounds (optional)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=1.0,
        key="conn_fare",
    )
    ticket = col_ticket.selectbox("Ticket type", ["single", "return"], key="conn_ticket")
    fare = fare_value if fare_value > 0 else None
    if wanted and not HHMM.match(wanted):
        st.warning("Write the time as hours and minutes, for example 10:06.")
        return
    try:
        status, body = client.connection_with_retry(rid, wanted or None, fare, ticket)
    except client.ApiDown:
        st.error("The risk service stopped answering. Please try again in a minute.")
        return
    if status == 202:
        st.info("We are still gathering live data for this train. Try again in a moment.")
        st.button("Try again", key="conn_retry")
        return
    if status == 404:
        st.info("We have no live information for this train right now.")
        return
    if status == 503:
        st.info("Connection forecasts are not available right now.")
        return
    if status == 504:
        st.info("That took too long. Please try again.")
        return
    if status != 200:
        st.error(f"Something went wrong (code {status}).")
        return
    _show_connection(body, fare)


def main() -> None:
    st.set_page_config(page_title="Train arrival range", page_icon="🚆", layout="centered")
    st.title("When will my train arrive?")

    try:
        routes = client.get_routes()
    except client.ApiDown:
        st.error("The risk service is not reachable right now. Please try again in a minute.")
        st.stop()
        return

    arrival_view, connection_view = st.tabs(["Train arrival", "My connection"])
    with arrival_view:
        arrival_tab(routes)
    with connection_view:
        connection_tab()


main()
