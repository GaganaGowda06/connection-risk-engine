import pandas as pd

from rail_risk.ingest.stations import match_stations, normalise_name


def test_normalise_name_drops_station_words_and_punctuation():
    assert normalise_name("Birmingham New Street Rail Station") == "birmingham new street"
    assert normalise_name("St. Pancras (Intl) Rail Station") == "st pancras"
    assert normalise_name("Bristol Temple Meads & Co") == "bristol temple meads and co"


def test_match_by_name_and_place_then_overlap_then_place_only_then_unmatched():
    naptan = pd.DataFrame(
        {
            "ATCOCode": ["9100AAAAAAA", "9100BBBBBBB", "9100CCCCCCC", "9100DDDDDDD", "490012345"],
            "CommonName": [
                "Alpha Rail Station",
                "Beta North Rail Station",
                "Gamma - Car Park",
                "Delta Rail Station",
                "A bus stop",
            ],
            "Latitude": [52.0, 53.0, 54.0, 55.0, 52.0],
            "Longitude": [-1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    stations = pd.DataFrame(
        {
            "stationName": ["Alpha", "Beta", "Gamma", "Elsewhere"],
            "lat": [52.0005, 53.0005, 54.0004, 60.0],
            "long": [-1.0, -1.0, -1.0, -1.0],
            "crsCode": ["ALP", "BET", "GAM", "ELS"],
        }
    )
    out = match_stations(naptan, stations).set_index("tiploc")
    assert list(out.index) == ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD"]  # bus stop ignored
    assert out.loc["AAAAAAA", ["crs", "method"]].tolist() == ["ALP", "name_and_place"]
    assert out.loc["BBBBBBB", ["crs", "method"]].tolist() == ["BET", "name_overlap_and_place"]
    assert out.loc["CCCCCCC", ["crs", "method"]].tolist() == ["GAM", "place_only"]
    assert out.loc["DDDDDDD", "method"] == "unmatched" and pd.isna(out.loc["DDDDDDD", "crs"])
