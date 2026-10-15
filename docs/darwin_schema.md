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
