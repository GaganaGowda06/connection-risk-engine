"""TIPLOC to CRS from two open files that need no account.

NaPTAN (DfT, Open Government Licence) lists every rail station with an ATCO code of 9100 plus its
TIPLOC, a name and a position. A public station list (davwheat/uk-railway-stations, built from the
stations Darwin can be queried for) gives name, position and CRS. The two are joined on name and
position, so no CRS is typed in by hand and the method is recorded per row.
"""

import re
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

NAPTAN_URL = "https://naptan.api.dft.gov.uk/v1/access-nodes?dataFormat=csv&atcoAreaCodes=910"
STATIONS_URL = "https://raw.githubusercontent.com/davwheat/uk-railway-stations/main/stations.json"
NEAR_M = 400.0
CLOSE_M = 150.0
_DROP = re.compile(r"\b(rail|railway|station|stn|sta|\(.*?\))\b|[^a-z0-9 ]")


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "rail-risk"}) as c:
        resp = c.get(url)
        resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def normalise_name(name: str) -> str:
    text = re.sub(r"\(.*?\)", " ", str(name).lower())
    text = _DROP.sub(" ", text.replace("&", " and "))
    return " ".join(text.split())


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def _tokens(name: str) -> set[str]:
    return set(normalise_name(name).split())


def match_stations(naptan: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """One row per NaPTAN TIPLOC with its CRS, how it was matched and the distance in metres.

    naptan needs ATCOCode, CommonName, Latitude, Longitude; stations needs stationName, lat, long,
    crsCode. A name match within NEAR_M wins, then a name overlap of at least half the words
    within NEAR_M, then the nearest station within CLOSE_M. Anything else stays unmatched.
    """
    nap = naptan[naptan["ATCOCode"].astype(str).str.startswith("9100")].copy()
    nap["tiploc"] = nap["ATCOCode"].astype(str).str[4:]
    nap = nap.drop_duplicates("tiploc").reset_index(drop=True)
    st = stations.reset_index(drop=True)
    dist = _haversine_m(
        nap["Latitude"].to_numpy(float),
        nap["Longitude"].to_numpy(float),
        st["lat"].to_numpy(float),
        st["long"].to_numpy(float),
    )
    st_norm = st["stationName"].map(normalise_name).to_numpy()
    st_tok = [set(n.split()) for n in st_norm]
    rows = []
    for i, rec in nap.iterrows():
        near = np.where(dist[i] <= NEAR_M)[0]
        norm = normalise_name(rec["CommonName"])
        toks = set(norm.split())
        pick, how = None, "unmatched"
        exact = [j for j in near if st_norm[j] == norm]
        if exact:
            pick, how = min(exact, key=lambda j: dist[i, j]), "name_and_place"
        else:
            overlap = [
                j for j in near if toks and len(toks & st_tok[j]) / len(toks | st_tok[j]) >= 0.5
            ]
            if overlap:
                pick, how = min(overlap, key=lambda j: dist[i, j]), "name_overlap_and_place"
            else:
                j = int(np.argmin(dist[i]))
                if dist[i, j] <= CLOSE_M:
                    pick, how = j, "place_only"
        rows.append(
            {
                "tiploc": rec["tiploc"],
                "crs": None if pick is None else st.at[pick, "crsCode"],
                "method": how,
                "dist_m": None if pick is None else round(float(dist[i, pick]), 1),
                "naptan_name": rec["CommonName"],
                "station_name": None if pick is None else st.at[pick, "stationName"],
            }
        )
    return pd.DataFrame(rows)


def build_station_map(workdir: Path) -> pd.DataFrame:
    nap = pd.read_csv(fetch(NAPTAN_URL, workdir / "naptan_910.csv"), low_memory=False)
    stations = pd.read_json(fetch(STATIONS_URL, workdir / "stations.json"))
    return match_stations(nap, stations)
