# Decisions

- 2026-10-14 Package name: rail_risk, src layout.
- 2026-10-14 Routes: BHM to EUS, BHM to MAN, and AST to EUS via BHM.
- 2026-10-14 Repo will be public on GitHub later.
- 2026-10-14 Terraform is plan-only, no AWS.
- 2026-10-14 Interval method: LightGBM quantile models with conformalized quantile regression; fallback empirical residual quantiles per route and hour.
- 2026-10-14 gitleaks via system binary plus full-history CI scan.
- 2026-10-14 Repo public on GitHub; commits use the noreply email.
- 2026-10-14 CI runners pinned to ubuntu-24.04 so the 2026-10-19 ubuntu-latest switch to Ubuntu 26 cannot break CI unannounced.
- 2026-10-15 Cancellation labelling unresolved: no cancelled service appeared in 25 sampled BHM to EUS services, so HSP cancellation representation is unknown. Reason codes are numeric strings on late stops. Needs a larger sample using one hour windows.
