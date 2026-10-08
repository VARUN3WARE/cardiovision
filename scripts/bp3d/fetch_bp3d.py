#!/usr/bin/env python3
"""Fetch only the BodyParts3D meshes CardioVision needs, without downloading the whole archive.

The official BodyParts3D v4.0 IS-A archive (isa_BP3D_4.0_obj_99.zip, ~143 MB, 2,234 OBJ files) is served
with HTTP range support. This script reads the zip's central directory over HTTP Range, then fetches and
inflates just the members listed in parts.json, verifying each one's CRC32. Standard library only.

    python3 scripts/bp3d/fetch_bp3d.py            # heart + thorax parts listed in parts.json
    python3 scripts/bp3d/fetch_bp3d.py --list     # print the archive's member table and exit

BodyParts3D, (c) The Database Center for Life Science, licensed under CC Attribution 4.0 International.
https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP_URL = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_BP3D_4.0_obj_99.zip"
CACHE = HERE / "cache"


def http_range(url: str, start: int, end: int | None = None, retries: int = 4) -> tuple[bytes, str]:
    """GET bytes [start, end] (inclusive; end=None means to EOF; start<0 means a suffix)."""
    rng = f"bytes={start}-" if end is None and start >= 0 else (f"bytes={start}" if start < 0 else f"bytes={start}-{end}")
    req = urllib.request.Request(url, headers={"Range": rng, "User-Agent": "cardiovision-bp3d-fetch/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored the Range header (HTTP {r.status})")
                return r.read(), r.headers.get("Content-Range", "")
        except Exception as e:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise AssertionError


def central_directory(url: str) -> dict[str, dict]:
    tail, content_range = http_range(url, -66000)
    total = int(content_range.rsplit("/", 1)[1])
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("end-of-central-directory record not found")
    _, _, _, _, n_entries, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if cd_offset >= total - len(tail):  # the directory is already inside the tail we fetched
        cd = tail[len(tail) - (total - cd_offset):][:cd_size]
    else:
        cd, _ = http_range(url, cd_offset, cd_offset + cd_size - 1)
    entries, p = {}, 0
    for _ in range(n_entries):
        (sig, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _, off) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        if sig != 0x02014B50:
            raise RuntimeError("corrupt central directory")
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        entries[name] = dict(method=method, crc=crc, csize=csize, usize=usize, offset=off)
        p += 46 + nlen + xlen + clen
    return entries


def fetch_member(url: str, e: dict) -> bytes:
    head, _ = http_range(url, e["offset"], e["offset"] + 29)
    sig, _, _, method, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", head)
    if sig != 0x04034B50:
        raise RuntimeError("bad local header")
    start = e["offset"] + 30 + nlen + xlen
    raw, _ = http_range(url, start, start + e["csize"] - 1)
    data = zlib.decompress(raw, -15) if method == 8 else raw
    if zlib.crc32(data) & 0xFFFFFFFF != e["crc"] or len(data) != e["usize"]:
        raise RuntimeError("CRC32 / size mismatch")
    return data


def wanted_ids(parts: dict, groups: list[str] | None) -> list[str]:
    ids: list[str] = []
    for gname, group in parts["groups"].items():
        if groups and gname not in groups:
            continue
        for item in group["parts"]:
            ids.extend(item["files"])
    return sorted(set(ids), key=lambda s: (int("".join(c for c in s[2:] if c.isdigit())), s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the archive member table and exit")
    ap.add_argument("--groups", nargs="*", help="only these groups from parts.json (default: all)")
    args = ap.parse_args()

    print(f"Reading central directory of {ZIP_URL}")
    entries = central_directory(ZIP_URL)
    by_id = {Path(n).stem: (n, e) for n, e in entries.items() if n.lower().endswith(".obj")}
    print(f"  {len(by_id)} OBJ members")
    if args.list:
        for fid, (n, e) in sorted(by_id.items()):
            print(f"{fid}\t{e['usize']}\t{e['csize']}\t{n}")
        return 0

    parts = json.loads((HERE / "parts.json").read_text())
    ids = wanted_ids(parts, args.groups)
    CACHE.mkdir(exist_ok=True)
    missing = [i for i in ids if i not in by_id]
    if missing:
        print(f"  not in archive: {' '.join(missing)}", file=sys.stderr)
        return 1
    got = cached = 0
    for fid in ids:
        name, e = by_id[fid]
        out = CACHE / f"{fid}.obj"
        if out.exists() and zlib.crc32(out.read_bytes()) & 0xFFFFFFFF == e["crc"]:
            cached += 1
            continue
        out.write_bytes(fetch_member(ZIP_URL, e))
        got += 1
        print(f"  {fid}  {e['usize'] / 1e6:6.2f} MB  ({e['csize'] / 1e6:.2f} MB transferred)")
    total = sum(by_id[i][1]["csize"] for i in ids) / 1e6
    print(f"Done: {got} fetched, {cached} already cached, {len(ids)} parts, {total:.1f} MB compressed in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
