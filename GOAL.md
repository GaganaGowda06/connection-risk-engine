# GOAL

## The user and the pain
UK rail passengers see "on time" until the last minute, then a vague delay or a missed connection. They need an honest arrival range, the chance of missing a connection, and what they are owed, for any journey, early enough to act.

## Scope (as Trainline would build it)
Any GB rail route and any operator, served live from the full Darwin feed. The current two-route model is a stepping stone, not the product.

## Pass/fail tests
All tests run on routes the model never trained on (leave-routes-out), unless stated.
1. Breadth: at least 20 routes, at least 5 operators, short, medium and long journeys, and at least one major hub interchange.
2. Accuracy: arrival MAE beats persistence and the historical baseline on held-out routes at 15 and 60 minutes out, with the paired 95% CI below zero.
3. Ranges: 80% arrival ranges cover 0.78 to 0.82 on held-out routes overall, with no held-out route below 0.70.
4. Late risk: P(delay at least 15 minutes) beats the historical baseline on Brier, with ECE at or below 0.05.
5. Connections: P(miss) beats the gap-bucket baseline on Brier at two or more hubs, with ECE at or below 0.05 on pairs whose inbound has not yet arrived.
6a. Out of time: the most recent archive week (2026-10-17 to 2026-10-23) is never trained, tuned or calibrated on, and is replayed through the service in Stage 6. Metrics on that week fall within the 95% CIs of tests 2 to 4.
6b. Live pipeline: our own capture runs on an always-on host with under 1% loss. Seven days is the target; if the project ends sooner, the README states the actual days and loss honestly.
7. Data quality: every route passes the same cleaning checks (types, timezones, station matching, planned closures told apart from missing data) with no route-specific code.
8. Trust: every number shown traces to a model output or a verified rule; unverified Delay Repay rules are never priced.
9. Reliability: p95 under 2 seconds, visible degradation when data is stale, automatic recovery after a restart.

## Not success
- Working only on routes the model trained on.
- Green CI without the leave-routes-out tests.
- A blocking limitation written up as a footnote.
- README numbers from smoke runs or familiar routes only.
- A capture that depends on a laptop staying awake.

## Rules for every stage
1. Open each stage report with a GOAL check: which tests pass, fail or are untested.
2. Break it on unseen routes, at scale and under failure before calling it done.
3. Research real systems first (papers, repos, engineering blogs, benchmarks and beyond); compare three options on a small sample.
4. If the plan cannot reach this goal, say so and bring the stronger fix; reopen any decision that blocks it.
5. Propose one stronger option per stage, unasked.
6. Use the right skills and tools automatically; ask first when unsafe or unclear.
7. After every stage, update PLAN.md and this file's last line, push, and remind the user to sync the GitHub repo in the project's Context panel (add PLAN.md there the first time it exists).

## Guards
Anchoring, passivity, overconfidence on familiar data, reflex apologies, missing ownership, reacting without the goal, dropping warnings for brevity, unresearched scope, drift, rationalising blockers.

## Done means
Tests 1 to 9 pass, with evidence from real runs and run ids, stated honestly in the README.

Current plan: see PLAN.md, now at Stage 1 (data foundation)
