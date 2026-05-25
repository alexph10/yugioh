from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Default fractional crop box for the art window on a standard new style
# Yu Gi Oh card. Tuned for ygoprodeck full size scans (~421x614). These
# defaults trim the name bar, type line, attribute icon, text box, and
# bottom matter while keeping a small margin of frame around the art.
DEFAULT_LEFT = 0.095
DEFAULT_TOP = 0.215
DEFAULT_RIGHT = 0.905
DEFAULT_BOTTOM = 0.625


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop the art panel out of full Yu Gi Oh card scans."
    )
    p.add_argument("--input-dir", type=str, default="data/raw/yugioh_cards")
    p.add_argument("--output-dir", type=str, default="data/processed/art")
    p.add_argument(
        "--manifest",
        type=str,
        default="data/raw/yugioh_cards/manifest.csv",
        help="Optional manifest CSV. Falls back to a directory scan if missing.",
    )
    p.add_argument(
        "--output-manifest",
        type=str,
        default="data/processed/art/manifest.csv",
    )
    p.add_argument(
        "--out-size",
        type=int,
        default=256,
        help="Final square size in pixels. Set to 0 to keep native crop size.",
    )
    p.add_argument("--left", type=float, default=DEFAULT_LEFT)
    p.add_argument("--top", type=float, default=DEFAULT_TOP)
    p.add_argument("--right", type=float, default=DEFAULT_RIGHT)
    p.add_argument("--bottom", type=float, default=DEFAULT_BOTTOM)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re crop files that already exist in the output directory.",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=200,
        help="Skip source images whose shortest side is below this many pixels.",
    )
    return p.parse_args()


def collect_inputs(input_dir: Path, manifest_csv: Path) -> list[Path]:
    if manifest_csv.is_file():
        paths: list[Path] = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("filename") or "").strip()
                if not name:
                    continue
                p = input_dir / name
                if p.is_file():
                    paths.append(p)
        if paths:
            return paths

    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def crop_one(
    src: Path,
    dst_dir: Path,
    box_fracs: tuple[float, float, float, float],
    out_size: int,
    min_size: int,
    overwrite: bool,
) -> tuple[bool, str, str]:
    dst = dst_dir / (src.stem + ".png")
    if dst.exists() and not overwrite and dst.stat().st_size > 0:
        return True, str(dst.name), "exists"

    try:
        with Image.open(src) as raw:
            im = raw.convert("RGB")
            w, h = im.size
            if min(w, h) < min_size:
                return False, src.name, f"too small ({w}x{h})"

            lf, tf, rf, bf = box_fracs
            box = (int(w * lf), int(h * tf), int(w * rf), int(h * bf))
            cw, ch = box[2] - box[0], box[3] - box[1]
            if cw <= 0 or ch <= 0:
                return False, src.name, "invalid crop box"

            # Force a square crop centered on the art window so resize does
            # not stretch portrait or landscape leftovers.
            side = min(cw, ch)
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            half = side // 2
            sq = (cx - half, cy - half, cx - half + side, cy - half + side)
            art = im.crop(sq)

            if out_size > 0:
                art = art.resize(
                    (out_size, out_size), Image.Resampling.LANCZOS
                )

            tmp = dst.with_suffix(dst.suffix + ".part")
            art.save(tmp, format="PNG", optimize=True)
            tmp.replace(dst)
            return True, dst.name, "ok"

    except (UnidentifiedImageError, OSError) as e:
        return False, src.name, f"{type(e).__name__}: {e}"


def main() -> int:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.is_dir():
        print(f"ERROR: input dir not found: {in_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_inputs(in_dir, Path(args.manifest))
    if not sources:
        print("Nothing to process.", file=sys.stderr)
        return 1
    print(f"Cropping {len(sources)} images -> {out_dir}", flush=True)

    box = (args.left, args.top, args.right, args.bottom)
    rows: list[dict[str, str]] = []
    ok = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                crop_one,
                src,
                out_dir,
                box,
                args.out_size,
                args.min_size,
                args.overwrite,
            ): src
            for src in sources
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            success, name, info = fut.result()
            if success:
                ok += 1
                rows.append({"filename": name, "source": futures[fut].name})
                if info == "exists":
                    skipped += 1
            else:
                failed += 1

            if i % 500 == 0 or i == len(sources):
                print(
                    f"[{i:>5}/{len(sources)}] ok={ok} skipped={skipped} failed={failed}",
                    flush=True,
                )

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Done. saved={ok} (already_existed={skipped}) failed={failed}",
        flush=True,
    )
    print(f"Output: {out_dir}", flush=True)
    print(f"Manifest: {out_manifest}", flush=True)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
