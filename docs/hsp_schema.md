# HSP response shapes

Based on one probe run on 2026-10-15 against the RDM HSP product, BHM to EUS, departures 07:00 to 08:00 on 2026-10-01. Raw responses were saved to data/probe, which is git-ignored. Everything below comes from those two responses only, so it is a sample, not a full specification.

## Requests that worked

Both calls returned HTTP 200 with the legacy body shapes and the `x-apikey` header.

- `POST {base}/serviceMetrics` with from_loc, to_loc, from_time, to_time, from_date, to_date and days. The path is confirmed.
- `POST {base}/serviceDetails` with a single `rid` taken from serviceMetrics.

## serviceMetrics response

Top-level keys: `header` and `Services`. Note the capital S.

- `header` holds `from_location` ("BHM") and `to_location` ("EUS") only.
- `Services` is a list. This probe returned 5 services.

Each element of `Services` has two keys.

`serviceAttributesMetrics` fields, all strings:
- `origin_location`: "BHM". Can be a station before the queried one, for example "WVH" and "MAN" both appeared.
- `destination_location`: "EUS".
- `gbtt_ptd`: "0706". Scheduled departure.
- `gbtt_pta`: "0920". Scheduled arrival.
- `toc_code`: "LM" or "VT".
- `matched_services`: "1".
- `rids`: a list of strings, for example ["202610018980803"]. One rid per service in this sample.

`Metrics` is a list with one object per tolerance setting, all strings except the flag:
- `tolerance_value`: "0"
- `num_not_tolerance`: "1"
- `num_tolerance`: "0"
- `percent_tolerance`: "0"
- `global_tolerance`: true (a JSON boolean)

The scheduled times appear to be at the queried from and to stations, not at the service origin. A service with origin "MAN" shows 0747 to 0906, which only makes sense as BHM to EUS. This is inferred, not stated by the API.

Only tolerance 0 appeared, so "tolerance" here looked like minutes of lateness at the destination, with 0 meaning on time or early. That is also inferred.

## serviceDetails response

Top-level key: `serviceAttributesDetails`, with `date_of_service` ("2026-10-01"), `toc_code` ("LM"), `rid` ("202610018980803") and `locations`.

`locations` is a list in running order, one per calling point (15 for this service, BHM to EUS). Every field is a string:
- `location`: CRS code, for example "BKW".
- `gbtt_ptd`: scheduled departure, empty at the final stop.
- `gbtt_pta`: scheduled arrival, empty at the first stop.
- `actual_td`: actual departure, empty at the final stop.
- `actual_ta`: actual arrival, empty at the first stop.
- `late_canc_reason`: reason code, empty for every location in this sample.

## Formats

- Times are four-digit HHMM strings in 24 hour form, with no colon, no seconds and no date. Missing values are empty strings, not null.
- Dates are ISO `YYYY-MM-DD` in `date_of_service` and in the request.
- The first eight digits of a rid are the service date as YYYYMMDD. The rest is an identifier. Treat rids as opaque strings.
- Whether times after midnight roll over or carry a day marker is not known from this sample.

## What the sample does and does not show

- Actual arrival times: yes. `actual_ta` is populated at every calling point that has a scheduled arrival.
- Cancellation flags: none seen. There is no cancelled field, and the one service returned ran in full. Cancelled services may be absent from the data, or marked by empty actual times or by `late_canc_reason`. This needs a sample that contains a cancellation.
- Delay reason codes: the `late_canc_reason` field exists but was empty at every location, even though the service reached EUS 6 minutes late (0926 against 0920). Its code format is not known.
