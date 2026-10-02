#!/usr/bin/env python3
"""Download and validate one natural-color shard plus a ScanQA annotation file."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


ANNOTATION_NAMES = {
    "train": "ScanQA_v1.0_train.json",
    "val": "ScanQA_v1.0_val.json",
    "test_w_obj": "ScanQA_v1.0_test_w_obj.json",
    "test_wo_obj": "ScanQA_v1.0_test_wo_obj.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download one benchmark shard ZIP and its ScanQA annotation.",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-url", required=True, help="Google Drive sharing URL of the shard ZIP.")
    annotation = parser.add_mutually_exclusive_group(required=True)
    annotation.add_argument("--annotation-url",
                            help="Google Drive sharing URL of the raw ScanQA JSON.")
    annotation.add_argument("--annotation-file", type=Path,
                            help="Existing local ScanQA JSON; validate it without downloading again.")
    parser.add_argument("--split", choices=tuple(ANNOTATION_NAMES), default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("downloads"))
    parser.add_argument("--redownload", action="store_true",
                        help="Replace files that already exist and validate the new downloads.")
    return parser.parse_args()


def shard_tag(index: int, total: int) -> str:
    if total < 1 or not 1 <= index <= total:
        raise ValueError("Require 1 <= --shard-index <= --num-shards")
    width = max(2, len(str(total)))
    return f"shard_{index:0{width}d}_of_{total:0{width}d}"


def download(url: str, destination: Path, redownload: bool) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not redownload:
        print(f"Reuse download: {destination}", flush=True)
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    command = [sys.executable, "-m", "gdown", url, "-O", str(temporary)]
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Download is missing or empty: {temporary}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def validate_shard(archive: Path, expected_tag: str) -> dict[str, int | str]:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Shard download is not a valid ZIP: {archive}")
    with zipfile.ZipFile(archive) as handle:
        files = [name.replace("\\", "/") for name in handle.namelist() if not name.endswith("/")]
    manifests = [name for name in files if name.endswith("/manifest.csv") or name == "manifest.csv"]
    images = [name for name in files if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    named_shards = sorted({part for name in files for part in name.split("/") if part.startswith("shard_")})
    if not manifests:
        raise ValueError(f"No manifest.csv found inside {archive}")
    if not images:
        raise ValueError(f"No images found inside {archive}")
    if named_shards and expected_tag not in named_shards:
        raise ValueError(f"Expected {expected_tag}, but ZIP contains shard directories: {named_shards[:10]}")
    return {"archive": str(archive), "files": len(files), "images": len(images),
            "manifests": len(manifests)}


def validate_annotation(path: Path) -> dict[str, int | str]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Annotation is not a valid JSON file: {path}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Annotation must be a non-empty JSON array: {path}")
    if not all(isinstance(row, dict) and row.get("question_id") for row in rows):
        raise ValueError(f"Annotation rows must contain question_id: {path}")
    return {"annotation": str(path), "questions": len(rows)}


def main() -> None:
    args = parse_args()
    tag = shard_tag(args.shard_index, args.num_shards)
    output_dir = args.output_dir.expanduser().resolve()
    archive = download(args.shard_url, output_dir / f"{tag}.zip", args.redownload)
    if args.annotation_file:
        annotation = args.annotation_file.expanduser().resolve()
        if not annotation.is_file():
            raise FileNotFoundError(f"Annotation not found: {annotation}")
    else:
        annotation = download(
            args.annotation_url, output_dir / ANNOTATION_NAMES[args.split], args.redownload,
        )
    report = {
        "status": "ready",
        "shard": validate_shard(archive, tag),
        "scanqa": validate_annotation(annotation),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nUse these local inputs:")
    print(f"  --scene-archive {archive}")
    print(f"  --annotation-file {annotation}")


if __name__ == "__main__":
    main()
