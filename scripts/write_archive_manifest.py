#!/usr/bin/env python3
"""Write data/manifests/darwin_archive.csv (path, size, sha256, gzip_ok) for every mirrored file.

Every file is read once in full: gzip -t equivalent integrity check plus the sha256, compared with
the hash recorded when it was fetched. The CSV is committed; the raw files never are.
"""

import csv
import hashlib
import sqlite3
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("data/raw/darwin_archive")
OUT = Path("data/manifests/darwin_archive.csv")
DB = Path("data/interim/archive_manifest.sqlite")


def check(path: str) -> tuple[str, int, str, bool]:
    p = Path(path)
    sha = hashlib.sha256()
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)
    ok = True
    with p.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            sha.update(chunk)
            try:
                inflate.decompress(chunk)
            except zlib.error:
                ok = False
                break
    ok = ok and inflate.eof
    return str(p.relative_to(ROOT)), p.stat().st_size, sha.hexdigest(), ok


def main() -> int:
    files = sorted(str(p) for p in ROOT.rglob("*.pport.gz"))
    recorded = {
        f"{d}/{h:02d}.pport.gz": s
        for d, h, s in sqlite3.connect(DB).execute(
            "SELECT day, hour, sha256 FROM files WHERE status='ok'"
        )
    }
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, files, chunksize=8))
    bad = [r for r in rows if not r[3]]
    mismatch = [r[0] for r in rows if recorded.get(r[0]) not in (None, r[2])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "size", "sha256", "gzip_ok"])
        w.writerows(rows)
    print(
        f"files={len(rows)} bytes={sum(r[1] for r in rows)} gzip_bad={len(bad)} "
        f"sha_mismatch={len(mismatch)} unrecorded={sum(r[0] not in recorded for r in rows)}"
    )
    for r in bad[:10]:
        print("BAD", r[0])
    return 1 if bad or mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
