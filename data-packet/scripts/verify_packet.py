#!/usr/bin/env python3
"""
verify_packet.py — confirm every file in MANIFEST.csv matches its sha256 and
loads without error. Run this after unpacking the data packet.

Usage:
    python3 scripts/verify_packet.py
    # or, from the data-packet root:
    cd data-packet && python3 scripts/verify_packet.py

Exit code 0 if all checks pass, 1 if any fail.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"ERROR: MANIFEST.csv missing at {MANIFEST}", file=sys.stderr)
        return 1

    failures: list[str] = []
    json_failures: list[str] = []
    n_checked = 0

    with MANIFEST.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["filename"]
            path = ROOT / rel
            if not path.exists():
                failures.append(f"MISSING: {rel}")
                continue

            actual_size = path.stat().st_size
            expected_size = int(row["size_bytes"])
            if actual_size != expected_size:
                failures.append(f"SIZE MISMATCH: {rel} (expected {expected_size}, got {actual_size})")
                continue

            actual_sha = sha256(path)
            expected_sha = row["sha256"]
            if actual_sha != expected_sha:
                failures.append(f"SHA MISMATCH: {rel}")
                continue

            if rel.endswith(".json"):
                try:
                    with path.open() as jf:
                        json.load(jf)
                except json.JSONDecodeError as exc:
                    json_failures.append(f"JSON PARSE: {rel} — {exc}")
                    continue

            n_checked += 1

    print(f"Verified {n_checked} files against MANIFEST.csv")
    if failures or json_failures:
        for line in failures + json_failures:
            print(line, file=sys.stderr)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
