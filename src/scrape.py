"""
Download up to 10,000 Yu-Gi-Oh! card images locally using the YGOPRODeck API.

Source:
- Metadata endpoint: https://db.ygoprodeck.com/api/v7/cardinfo.php
- Image URLs come from each card's `card_images` array

Notes:
- The API guide says to download/store locally and not hotlink images.
- Be respectful with request rate. This script uses a conservative worker count.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
OUTPUT_DIR = Path("yugioh_cards_10000")
IMAGES_DIR = OUTPUT_DIR / "data/raw/yugioh_cards"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"

TARGET_COUNT = 10_000
MAX_WORKERS = 8
REQUEST_TIMEOUT = 30
RETRIES = 4
BACKOFF_BASE = 1.5

session = requests.Session()
session.headers.update(
    {"User-Agent": "Mozilla/5.0 (compatible; yugioh-image-downloader/1.0)"}
)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "card"


def fetch_all_cards() -> List[dict]:
    r = session.get(API_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["data"]


def build_download_list(cards: List[dict], target_count: int) -> List[Dict]:
    items = []
    seen_urls = set()

    for card in cards:
        card_name = card.get("name", "unknown")
        base_id = str(card.get("id", "unknown"))

        for idx, image_info in enumerate(card.get("card_images", []), start=1):
            image_url = image_info.get("image_url")
            image_id = str(image_info.get("id", base_id))

            if not image_url or image_url in seen_urls:
                continue

            seen_urls.add(image_url)

            # If a card has alt artworks, keep them distinct.
            alt_suffix = f"-alt{idx}" if idx > 1 else ""
            filename = f"{image_id}{alt_suffix}-{slugify(card_name)}.jpg"

            items.append(
                {
                    "card_name": card_name,
                    "card_id": base_id,
                    "image_id": image_id,
                    "image_index": idx,
                    "image_url": image_url,
                    "filename": filename,
                }
            )

            if len(items) >= target_count:
                return items

    return items


def download_one(item: Dict) -> Tuple[bool, Dict, str]:
    dest = IMAGES_DIR / item["filename"]

    if dest.exists() and dest.stat().st_size > 0:
        return True, item, "exists"

    last_error = ""
    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(
                item["image_url"], timeout=REQUEST_TIMEOUT, stream=True
            ) as r:
                r.raise_for_status()
                content_type = r.headers.get("Content-Type", "")
                if "image" not in content_type.lower():
                    raise ValueError(f"Unexpected content type: {content_type}")

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            if dest.stat().st_size == 0:
                raise ValueError("Downloaded file is empty")

            return True, item, "downloaded"

        except Exception as e:
            last_error = str(e)
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            time.sleep(BACKOFF_BASE**attempt)

    return False, item, last_error


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching card metadata...")
    cards = fetch_all_cards()
    print(f"Total cards returned by API: {len(cards)}")

    items = build_download_list(cards, TARGET_COUNT)
    print(f"Prepared {len(items)} image URLs for download.")

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "card_name",
                "card_id",
                "image_id",
                "image_index",
                "image_url",
                "filename",
                "status",
            ],
        )
        writer.writeheader()

    with open(FAILURES_CSV, "w", newline="", encoding="utf-8") as ff:
        writer = csv.DictWriter(
            ff,
            fieldnames=[
                "card_name",
                "card_id",
                "image_id",
                "image_index",
                "image_url",
                "filename",
                "error",
            ],
        )
        writer.writeheader()

    completed = 0
    success_count = 0
    failure_count = 0

    print(f"Starting downloads with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, item): item for item in items}

        for future in as_completed(futures):
            ok, item, message = future.result()
            completed += 1

            if ok:
                success_count += 1
                with open(MANIFEST_CSV, "a", newline="", encoding="utf-8") as mf:
                    writer = csv.DictWriter(
                        mf,
                        fieldnames=[
                            "card_name",
                            "card_id",
                            "image_id",
                            "image_index",
                            "image_url",
                            "filename",
                            "status",
                        ],
                    )
                    row = dict(item)
                    row["status"] = message
                    writer.writerow(row)
            else:
                failure_count += 1
                with open(FAILURES_CSV, "a", newline="", encoding="utf-8") as ff:
                    writer = csv.DictWriter(
                        ff,
                        fieldnames=[
                            "card_name",
                            "card_id",
                            "image_id",
                            "image_index",
                            "image_url",
                            "filename",
                            "error",
                        ],
                    )
                    row = dict(item)
                    row["error"] = message
                    writer.writerow(row)

            if completed % 100 == 0 or completed == len(items):
                print(
                    f"Progress: {completed}/{len(items)} | "
                    f"ok={success_count} fail={failure_count}"
                )

    print("\nDone.")
    print(f"Downloaded: {success_count}")
    print(f"Failed:     {failure_count}")
    print(f"Images dir: {IMAGES_DIR.resolve()}")
    print(f"Manifest:   {MANIFEST_CSV.resolve()}")
    print(f"Failures:   {FAILURES_CSV.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
