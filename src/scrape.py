from __future__ import annotations

import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any

import requests


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    for candidate in [current.parent, *current.parents]:
        if (candidate / "src").exists():
            return candidate
    return current.parent


REPO_ROOT = find_repo_root(Path(__file__))
API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "yugioh_cards"
IMAGES_DIR = OUTPUT_DIR
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"

TARGET_COUNT = 10_000
MAX_WORKERS = 8
REQUEST_TIMEOUT = (5, 30)
RETRIES = 4
BACKOFF_BASE = 1.5
# ygoprodeck asks clients to rate-limit themselves to ~20 req/sec.
PER_REQUEST_DELAY = 0.06

_thread_state = local()


def get_session() -> requests.Session:
    if not hasattr(_thread_state, "session"):
        s = requests.Session()
        s.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; yugioh-image-downloader/1.0)"}
        )
        _thread_state.session = s
    return _thread_state.session


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "card"


def fetch_all_cards() -> list[dict[str, Any]]:
    print(f"Fetching card metadata from {API_URL} ...", flush=True)
    r = get_session().get(API_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    cards = data.get("data", [])
    print(f"Received metadata for {len(cards):,} cards.", flush=True)
    return cards


def build_download_list(
    cards: list[dict[str, Any]], target_count: int
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for card in cards:
        card_name = card.get("name", "unknown")
        base_id = str(card.get("id", "unknown"))

        for idx, image_info in enumerate(card.get("card_images", []), start=1):
            image_url = image_info.get("image_url")
            image_id = str(image_info.get("id", base_id))

            if not image_url or image_url in seen_urls:
                continue

            seen_urls.add(image_url)
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


def download_one(item: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    dest = IMAGES_DIR / item["filename"]

    if dest.exists() and dest.stat().st_size > 0:
        return True, item, "exists"

    last_error = ""
    session = get_session()

    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(
                item["image_url"], timeout=REQUEST_TIMEOUT, stream=True
            ) as r:
                r.raise_for_status()
                content_type = r.headers.get("Content-Type", "")
                if "image" not in content_type.lower():
                    raise ValueError(f"Unexpected content type: {content_type}")

                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if tmp.stat().st_size == 0:
                    tmp.unlink(missing_ok=True)
                    raise ValueError("Downloaded file is empty")

                tmp.replace(dest)

            time.sleep(PER_REQUEST_DELAY)
            return True, item, "downloaded"

        except Exception as e:
            last_error = str(e)
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            if attempt < RETRIES:
                time.sleep(BACKOFF_BASE**attempt)

    return False, item, last_error


def write_manifest(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "card_name",
        "card_id",
        "image_id",
        "image_index",
        "image_url",
        "filename",
        "status",
    ]
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_failures(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "card_name",
        "card_id",
        "image_id",
        "image_index",
        "image_url",
        "filename",
        "error",
    ]
    with open(FAILURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        cards = fetch_all_cards()
    except requests.RequestException as e:
        print(f"ERROR: failed to fetch card list: {e}", file=sys.stderr)
        return 1

    items = build_download_list(cards, TARGET_COUNT)
    print(
        f"Prepared {len(items):,} image targets (goal: {TARGET_COUNT:,}).",
        flush=True,
    )
    if not items:
        print("Nothing to download.", file=sys.stderr)
        return 1

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    downloaded = 0
    skipped = 0
    total = len(items)
    started = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(download_one, it): it for it in items}
        for i, fut in enumerate(as_completed(future_map), start=1):
            ok, item, info = fut.result()
            if ok:
                item = {**item, "status": info}
                successes.append(item)
                if info == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            else:
                failures.append({**item, "error": info})

            if i % 50 == 0 or i == total:
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{i:>5}/{total}] downloaded={downloaded} "
                    f"skipped={skipped} failed={len(failures)} "
                    f"({rate:.1f} img/s)",
                    flush=True,
                )

    write_manifest(successes)
    if failures:
        write_failures(failures)

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.1f}s. "
        f"saved={len(successes)} (new={downloaded}, existing={skipped}) "
        f"failed={len(failures)}",
        flush=True,
    )
    print(f"Images: {IMAGES_DIR}", flush=True)
    print(f"Manifest: {MANIFEST_CSV}", flush=True)
    if failures:
        print(f"Failures: {FAILURES_CSV}", flush=True)

    return 0 if len(successes) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
