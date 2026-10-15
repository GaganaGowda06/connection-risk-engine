# HSP response shapes

Based on two probe runs on 2026-10-15 against the RDM HSP product, BHM to EUS. Run one covered departures 07:00 to 08:00 on 2026-10-01 (5 services, 1 detail). Run two covered 06:00 to 10:00 on 2026-10-08 (20 services, 2 details). Raw responses were saved to data/probe, which is git-ignored. Everything below comes from those responses only, so it is a small sample, not a full specification.

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

## Reason codes

`late_canc_reason` holds a numeric string when populated. Two codes were seen, both on 2026-10-08:

- `"895"` at the final stop (EUS) of service 202610088734019 (VT, scheduled 0731, actual 0742, 11 minutes late). The earlier stops were empty although the train was already 5 to 6 minutes late from BHM onward.
- `"520"` on service 202610086717977 (LM) at five consecutive stops from RUG to EUS, including the final stop (scheduled 0821, actual 0839, 18 minutes late). The code repeats at each stop after the one where it was first attached.

So a code is attached at a stop and may repeat at later stops. A code appears at some late stops and not at others, so an empty value does not prove the train was on time. What the codes mean was not decoded here. They look like Darwin late-running reason codes and would need the Darwin reference table to translate.

## Cancellations

No cancelled service or cancelled stop was seen. Across the 25 services in the sample, every service returned by serviceMetrics had a populated serviceDetails when requested (3 requested), and every one ran from origin to EUS. Nothing in either response has a cancelled flag. How a cancellation is represented is therefore unresolved: it could be a missing service, a service with empty actuals and a reason code, or something else.

One pattern that is not a cancellation: service 202610088734019 has a scheduled arrival at SAD (0552) with an empty `actual_ta`, but the train then called at every later stop. An empty actual at a stop is a reporting gap, not evidence the train was cancelled.

## Metrics fields as observed

- `tolerance_value` was "0" in every row, so only one tolerance was returned. The request body had no tolerance field, and whether one can be requested was not tested.
- `num_tolerance` and `num_not_tolerance` were always "1" and "0", or "0" and "1", and `percent_tolerance` was "100" or "0". With one run per service, each service is simply on time or not, judged at the destination.
- On 2026-10-08, all 20 services had `percent_tolerance` "0". The service with the 18 minute delay and the one with the 11 minute delay both show 0, so the field does not rank how late a train was.
- `matched_services` was "1" and `rids` had one entry for all 25 services.

## Practical limits seen

- A four hour window (0600 to 1000) returned 20 services within the 30 second timeout. The next two four hour windows on the same day (1000 to 1400 and 1400 to 1800) each timed out twice. One hour windows returned quickly. Wide windows are unreliable.
- Time values never crossed midnight in this sample (earliest 0542, latest 0839), so rollover behaviour is still unknown.

## What the sample does and does not show

- Actual arrival times: yes. `actual_ta` is populated at nearly every calling point that has a scheduled arrival.
- Cancellation flags: none seen, see above.
- Delay reason codes: seen, as numeric strings, on late services only. Meaning not decoded.
