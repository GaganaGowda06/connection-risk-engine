# Darwin Push Port feed notes

Based on one trial capture on 2026-10-15, from 11:21 to 11:31 UK time (Thursday, daytime), on the topic `prod-1010-Darwin-Train-Information-Push-Port-IIII2_0-JSON`. The capture is 17,394 messages: a 24 second connection test of 665 messages plus the 10 minute run of 16,729. Everything below comes from those messages, so it is a ten minute daytime sample, not a full day.

## Volume

- Rate over the 10 minute run: 1,673 messages and 2.72 MB per minute, as delivered by Kafka.
- Projected per day, assuming the rate is flat: about 2.4 million messages and 3.9 GB uncompressed. Nights are quieter, so this is probably an overestimate.
- The stored gzip JSONL is about 14 times smaller than the delivered bytes, so a full unfiltered day would be roughly 0.3 GB on disk.
- Each Kafka message is a JMS wrapper (`destination`, `messageID`, `deliveryMode`, `timestamp` and so on). The Darwin payload is a JSON string in the wrapper's `bytes` field, with its own `ts`, `version` ("18.0") and `uR` object. The wrapper roughly doubles the size, so a long run should store only the parsed payload.

## Message types

Types are the keys of `uR`, apart from `updateOrigin`, `requestSource` and `requestID`. A message can hold more than one type.

| Type | Share of messages |
|---|---|
| TS (train status: forecasts and actuals) | 88.3% |
| FailureResp heartbeat responses (`code` "HBOK") | 5.6% |
| `uR` with no payload type | 1.7% |
| schedule | 1.7% |
| deactivated | 1.6% |
| association | 1.5% |
| formationLoading | 1.0% |
| scheduleFormations, serviceLoading, OW (station message) | under 0.2% together |

## Our four stations

The feed uses TIPLOC codes, not CRS codes, and contains no station names. These are the codes found in the data that match our stations by name, with the number of messages that mention each:

| Station | TIPLOC found | Messages |
|---|---|---|
| Birmingham New Street | BHAMNWS | 142 |
| London Euston | EUSTON | 100 |
| Manchester Piccadilly | MNCRPIC | 132 |
| Aston | ASTON | 12 |

The match is by name only. Other codes share the prefix (BHAMINT, BHAMMRS, MNCRDGT, MNCROXR and others) and are different stations or junctions. Co-occurrence supports the pairing but is thin: 2 messages carry both BHAMNWS and EUSTON, 8 carry BHAMNWS and MNCRPIC, and 4 carry ASTON and BHAMNWS. A reference table of TIPLOC to CRS would remove the doubt.

Filter rate: 372 of 17,394 messages (2.1%) mention at least one of these four TIPLOCs. Those messages are larger than average, at 3.3% of the bytes. Of them, 359 are TS, 13 are schedule and 13 are association.

## Projected volume with the route filter

- Messages: about 51,000 per day.
- Bytes: about 130 MB per day uncompressed, so well under 0.02 GB compressed.

A filter on the four TIPLOCs would keep through-services that call at our stations. It would miss a service that matters for a connection but never mentions them, which this sample cannot size.

## How cancellations appear

Cancellations are in `schedule` messages, not in TS. Three markers were seen.

- `cancelReason` on the schedule, a numeric string such as "832". It appeared on 64 of 296 schedule messages, which is 21 distinct services out of 403 distinct scheduled services in the sample.
- `can: "true"` on individual stops, inside the location lists `OR` (origin), `IP` (intermediate), `PP` (pass), `DT` (destination) and their `OP...` variants. Counts: IP 480, PP 128, DT 46, OR 44.
- `isCancelled: "true"` on an `association` (a join or split between two services).

A real fragment from a schedule message, where the last stops of a service are cancelled and the service carries a reason:

```
"DT":[{"tpl":"SPRNGBN","act":"TF","can":"true","pta":"11:14","wta":"11:14"}],"cancelReason":"832"
```

Cancellation is partial as well as total. In the example above the earlier stops run and only the later stops carry `can`. Only 3 of the 21 cancelled services had `can` on the origin (`OR`), which is the best sign of a fully cancelled train. This is read from the field names and values, and the schema meaning was not checked against the Darwin documentation.

## How delay reasons appear

Delay reasons are in TS messages as `LateReason`, a sibling of `Location` under `TS`. Two forms were seen:

- A bare code string: `"LateReason":"871"` (1,602 of 1,608 occurrences).
- An object with a location: `"LateReason":{"tiploc":"SWRBBDG","near":"true","":"579"}` (6 occurrences), where the empty-string key holds the code.

The most common codes were 895, 871, 824, 832, 841 and 816. Code 895 also appeared as `late_canc_reason` in the HSP sample, so the two feeds appear to share one reason code table. The cancel reasons in this sample were 832 (71 times), 911, 574 and 825, and 832 also appears as a late reason. The meaning of each code is not decoded here.

## Other format notes

- Times are `HH:MM` or `HH:MM:SS` strings in UK local time, with the service date in `ssd` ("2026-10-15"). The message `ts` carries a UTC offset ("+01:00").
- A service is identified by `rid` (15 digits starting with the service date) and `uid`.
- Forecasts and actuals sit under each TS `Location` as `arr`, `dep` or `pass`, with `et` (estimated time) and `src`. Actual times use `at` instead of `et`, as in `"pass":{"at":"11:21","atClass":"Automatic","src":"TD"}`. 7,552 of the 17,394 messages (43%) contain at least one `at`, so actuals are plentiful in the live feed.
- A `Location` is a list when there are several stops and a single object when there is one. A parser must accept both.

## Stored line format

The trial lines in `data/raw/darwin/2026-10-15/10.jsonl.gz` (received 10:21 to 10:31 UTC) use the first format: `received_at`, `partition`, `offset` and `value`, where `value` is the full JMS wrapper and the Darwin payload is the JSON string in `value.bytes`. Lines written by the resumable capture use `partition`, `offset`, `kafka_ts` (milliseconds) and `payload` (the parsed Darwin message), or `raw` when a message was not in the expected shape. Both formats share one file for that hour, and a reader must handle both.

After stripping the wrapper the feed is about 1.2 MB per minute, against 2.7 MB with the wrapper.

## Parsed tables

`rail-risk parse-darwin` reads each closed hourly file once and writes three Parquet tables under `data/interim/darwin/`, partitioned by `date=YYYY-MM-DD`, one part file per raw file and chunk (named like `2026-10-15T10_0.parquet`). Nothing is filtered to our routes. All timestamps are timezone-aware UTC, converted from Europe/London with the same rules as HSP (see DECISIONS.md and `rail_risk.ingest.times`). A file is parsed again only if it has grown since, for example after a resume appended to it. The file for the current UTC hour is skipped.

### darwin_schedules

One row per rid per raw file, keeping the latest copy when a schedule was sent more than once in that hour. A rid can therefore appear in more than one part file, and the row with the newest `message_ts` wins. Columns: `rid`, `uid`, `ssd` (service date), `train_id`, `toc`, `origin_tpl`, `dest_tpl`, `cancel_reason` and `cancel_reason_tpl` (the schedule-level `cancelReason`, a bare code or an object), `n_stops`, `n_cancelled_stops`, `stops`, `message_ts`, `partition` and `offset`.

`stops` is the ordered list of stops. Each has `seq`, `tpl`, `kind` (OR, OPOR, IP, OPIP, PP, OPDT or DT), `act`, `can` (true when the stop is cancelled) and the planned times `wta`, `pta`, `wtp`, `wtd` and `ptd` (working and public, arrival, pass and departure). The JSON groups stops by tag, which loses the running order, so the order is rebuilt from the times since the origin. A divided train has several DT stops, and `dest_tpl` is the one with the latest time. When a reader loads `stops` through pandas the nested timestamps lose their zone, but the Parquet type is `timestamp[us, tz=UTC]`.

### darwin_events

One row per TS location update and event type. Columns: `rid`, `uid`, `ssd`, `tiploc`, `event_type` (arr, dep or pass), `time_kind` (actual when `at` was given, forecast when `et` was given, otherwise unknown), `time`, `planned_time`, `src`, `delayed`, `late_reason` and `late_reason_tpl` (the message-level `LateReason`, as a bare code or an object, attached to every event of that message), `message_ts`, `partition` and `offset`. `planned_time` is the public time when present and the working time otherwise.

Times are placed on the day nearest the planned time, so a forecast just after midnight for a train planned before it lands on the next day. With no planned time the message time is the anchor. A forecast can be well before the plan: one stabled unit showed a forecast departure 2.5 hours ahead of its plan, and that is the data, not a parse error.

### darwin_associations

One row per association update. Columns: `category` (JJ, VV, LK or NP), `tiploc`, `main_rid`, `assoc_rid`, `is_cancelled`, `is_deleted`, `message_ts`, `partition` and `offset`. These are partitioned by the date of `message_ts`, since an association has no service date.

### Quarantine and counts

Rows that fail parsing or validation go to `darwin_<table>_quarantine/` with `partition`, `offset`, `quarantine_reason` and `raw`. Unreadable lines, unparsed payloads and messages with no timestamp go to `darwin_messages_quarantine/`. Rows are never dropped silently. Duplicates are lines with a repeated (partition, offset) within one raw file and are counted, then removed. A duplicate that falls in a different raw file is not removed here, so a consumer should deduplicate on (partition, offset) if it matters.

A truncated gzip member keeps every complete line, and the partial last line is counted as bytes lost in the manifest and in `rail-risk status`.

### First parse

The two closed files at the time of writing (2026-10-15 hours 10 and 11 UTC) took 3.1 s and 4.6 s. Together they held 3,740 schedule rows (3,380 distinct rids), 763,588 events and 2,756 associations, with no quarantined rows, no duplicates and no bytes lost. Of the 3,380 distinct schedules, 125 carry a `cancelReason` and 33 have every stop cancelled. 249 touch BHAMNWS, EUSTON, MNCRPIC or ASTON, and 5 of those have a cancelled stop at one of those four.
