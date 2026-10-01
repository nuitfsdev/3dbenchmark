#!/usr/bin/env python3
"""Benchmark one ZIP shard of ScanQA instance-mesh overview images with Qwen3-VL.

The official model corresponding to the commonly requested "Qwen3-VL 7B" size is
``Qwen/Qwen3-VL-8B-Instruct``.  A different/local checkpoint can be supplied with
``--model-id``.

Expected image layout inside each ZIP (extra parent directories are allowed)::

    shard_01_of_10/
      scene0011_00/
        val-scene0011-0/
          overview.png

``manifest.csv`` is optional because one fixed ``overview.png`` is joined to the
ScanQA annotation by its parent directory (the question ID).

The process is interruption-safe.  Every completed question is appended and fsynced
immediately.  Re-running the same command resumes unfinished questions.  Each shard
writes a merge-ready full JSONL plus a small raw-prediction JSONL, CSV, errors and a
summary.  Reference answers are stored for later global scoring but never sent to the
model.

Example::

    python benchmarks/benchmark_qwen3vl_8b_overview_shard.py \
      --shard-index 1 --num-shards 10 \
      --shard-url "https://drive.google.com/file/d/FILE_ID/view" \
      --annotation-url "https://drive.google.com/file/d/ANNOTATION_ID/view"

Required packages (install a CUDA-compatible torch separately)::

    pip install "transformers>=4.57.0" accelerate gdown pillow tqdm

Full precision is the default: BF16 on supported GPUs, otherwise FP16.  NF4 is
available only when explicitly requested with ``--load-in-4bit`` and requires
``bitsandbytes``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODEL_ID_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT_VERSION = "scanqa_instance_mesh_overview_single_answer_v1"
SCHEMA_VERSION = 1
ANNOTATION_FILENAMES = {
    "train": "ScanQA_v1.0_train.json",
    "val": "ScanQA_v1.0_val.json",
    "test_w_obj": "ScanQA_v1.0_test_w_obj.json",
    "test_wo_obj": "ScanQA_v1.0_test_wo_obj.json",
}
SYSTEM_PROMPT = """Answer one ScanQA question from one high-angle instance-mesh overview of an indoor scene.
The image contains textured object meshes selected for the question. Floors and visibility-safe wall fragments
may appear only as spatial context. Black text labels name object classes; each label belongs to the mesh nearest
to it. The overview can be partial, so use only clearly visible evidence and never invent hidden objects.

Reason silently in this order:
1. Identify exactly what the question asks: object, attribute, relation, location, or count.
2. Inspect mesh geometry and texture. Labels help identify classes but their screen positions do not establish
   spatial relations. For color, inspect the mesh texture rather than the black label box or walls/floor.
3. For a count, count each distinct relevant mesh once. Repeated class labels refer to separate objects only when
   attached to separate meshes. Do not count unlabeled wall/floor fragments unless explicitly asked.
4. Choose one canonical, shortest answer supported by the image.

Return exactly ONE short plain-text answer. For counts return only the number. Do not return an array, alternatives,
a sentence, reasoning, brackets, object IDs, or coordinate notation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run Qwen3-VL on exactly one ZIP shard of overview.png images.",
    )
    parser.add_argument("--shard-index", type=int, required=True,
                        help="One-based shard number, matching exporter shard_XX_of_YY.")
    parser.add_argument("--num-shards", type=int, required=True,
                        help="Total shard count used when the images were exported.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shard-url", help="Public Google Drive sharing URL of this shard ZIP.")
    source.add_argument("--shard-archive", type=Path, help="Already-downloaded shard ZIP.")
    source.add_argument("--data-dir", type=Path, help="Already-extracted shard directory.")
    parser.add_argument("--annotation-file", type=Path,
                        help="Local ScanQA JSON. Otherwise search the shard, then use --annotation-url.")
    parser.add_argument("--annotation-url", help="Google Drive sharing URL of the ScanQA JSON or ZIP.")
    parser.add_argument("--split", default="val", choices=tuple(ANNOTATION_FILENAMES))
    parser.add_argument("--scene-id", action="append", dest="scene_ids",
                        help="Optional scene filter; repeat for multiple scenes.")
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark_work/qwen3vl_overview_shards"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3vl_overview_shards"))
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to NF4 quantization. The default is full BF16/FP16 precision.")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-questions", type=int, help="Limit this shard for a smoke test.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Download/extract/validate the shard without loading the model.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete this shard's existing benchmark artifacts before running.")
    parser.add_argument("--redownload", action="store_true", help="Download the Drive ZIP again.")
    parser.add_argument("--overwrite-extract", action="store_true",
                        help="Delete and re-extract this shard's extraction directory.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 1 <= args.shard_index <= args.num_shards:
        raise ValueError("Require 1 <= --shard-index <= --num-shards")
    if args.max_new_tokens < 1 or args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Invalid token or pixel limits")
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be positive")


def shard_tag(args: argparse.Namespace) -> str:
    width = max(2, len(str(args.num_shards)))
    return f"shard_{args.shard_index:0{width}d}_of_{args.num_shards:0{width}d}"


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_").lower()


def run_checked(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def download_drive(url: str, destination: Path, redownload: bool = False) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not redownload:
        print(f"Reuse download: {destination}", flush=True)
        return destination
    if destination.exists():
        destination.unlink()
    run_checked([sys.executable, "-m", "gdown", "--fuzzy", url, "-O", str(destination)])
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Google Drive download is missing or empty: {destination}")
    return destination


def safe_extract_zip(archive: Path, target: Path, overwrite: bool = False) -> Path:
    archive = archive.expanduser().resolve()
    target = target.expanduser().resolve()
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Shard archive must be an existing ZIP: {archive}")
    marker = target / ".extract_complete.json"
    signature = {"archive": str(archive), "size": archive.stat().st_size,
                 "mtime_ns": archive.stat().st_mtime_ns}
    if overwrite and target.exists():
        shutil.rmtree(target)
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == signature:
                print(f"Reuse extraction: {target}", flush=True)
                return target
        except (OSError, json.JSONDecodeError):
            pass
        raise RuntimeError(
            f"Extraction directory already exists for different/incomplete data: {target}. "
            "Use --overwrite-extract to replace it."
        )
    target.mkdir(parents=True, exist_ok=True)
    target_prefix = str(target) + os.sep
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            destination = (target / member.filename).resolve()
            if destination != target and not str(destination).startswith(target_prefix):
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        handle.extractall(target)
    marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return target


def prepare_shard_data(args: argparse.Namespace, tag: str) -> Path:
    if args.data_dir:
        root = args.data_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory not found: {root}")
        return root
    if args.shard_archive:
        archive = args.shard_archive
    else:
        archive = args.work_dir / "downloads" / f"{tag}.zip"
        archive = download_drive(args.shard_url, archive, args.redownload)
    return safe_extract_zip(archive, args.work_dir / "extracted" / tag, args.overwrite_extract)


def select_shard_scope(root: Path, tag: str) -> Path:
    if root.name.lower() == tag.lower():
        return root
    matches = sorted({path.resolve() for path in root.rglob(tag) if path.is_dir()})
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple directories named {tag} under {root}: {matches[:5]}")
    other_shards = sorted(path for path in root.rglob("shard_*_of_*") if path.is_dir())
    if other_shards:
        raise FileNotFoundError(f"Requested {tag}, but archive contains: {[p.name for p in other_shards[:10]]}")
    return root


def discover_overviews(scope: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(scope.rglob("overview.png")):
        relative = path.relative_to(scope)
        if any(part.startswith(".") for part in relative.parts):
            continue
        question_id = path.parent.name
        if question_id in result:
            duplicates.setdefault(question_id, [str(result[question_id])]).append(str(path.resolve()))
        else:
            result[question_id] = path.resolve()
    if duplicates:
        examples = dict(list(duplicates.items())[:3])
        raise ValueError(f"More than one overview.png for the same question ID: {examples}")
    if not result:
        raise FileNotFoundError(f"No <question_id>/overview.png found under {scope}")
    return result


def prepare_annotation(args: argparse.Namespace, data_root: Path) -> Path:
    expected_name = ANNOTATION_FILENAMES[args.split]
    if args.annotation_file:
        path = args.annotation_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Annotation not found: {path}")
        return path
    found = sorted(data_root.rglob(expected_name))
    if len(found) == 1:
        return found[0].resolve()
    if len(found) > 1:
        print(f"Multiple annotations found; using {found[0]}", file=sys.stderr, flush=True)
        return found[0].resolve()
    if not args.annotation_url:
        raise FileNotFoundError(
            f"{expected_name} is not inside the shard. Supply --annotation-file or --annotation-url."
        )
    # Keep a .zip suffix so the same Drive URL may point either to a ZIP or to
    # the raw JSON; JSON parsing is content-based and does not require .json.
    destination = args.work_dir / "annotations" / f"{args.split}_annotation_payload.zip"
    downloaded = download_drive(args.annotation_url, destination, args.redownload)
    if zipfile.is_zipfile(downloaded):
        extracted = safe_extract_zip(downloaded, args.work_dir / "annotations" / args.split,
                                     args.overwrite_extract)
        matches = sorted(extracted.rglob(expected_name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {expected_name} in annotation ZIP; found {len(matches)}")
        return matches[0].resolve()
    return downloaded.resolve()


def number_suffix(question_id: str) -> int | str:
    match = re.search(r"-(\d+)$", question_id)
    return int(match.group(1)) if match else question_id


def load_eval_items(annotation_path: Path, overview_by_qid: dict[str, Path],
                    scene_ids: list[str] | None, max_questions: int | None
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_qid: dict[str, dict[str, Any]] = {}
    duplicate_annotations: set[str] = set()
    for row in annotations:
        question_id = str(row["question_id"])
        if question_id in by_qid:
            duplicate_annotations.add(question_id)
        by_qid[question_id] = row
    if duplicate_annotations:
        raise ValueError(f"Duplicate question IDs in annotation: {sorted(duplicate_annotations)[:10]}")
    wanted_scenes = set(scene_ids or [])
    unknown_images = sorted(set(overview_by_qid) - set(by_qid))
    items = []
    scene_mismatches = []
    for question_id, image_path in overview_by_qid.items():
        row = by_qid.get(question_id)
        if row is None or (wanted_scenes and row.get("scene_id") not in wanted_scenes):
            continue
        scene_id = str(row["scene_id"])
        if scene_id not in image_path.parts:
            scene_mismatches.append((question_id, scene_id, str(image_path)))
        items.append({
            "question_id": question_id,
            "scene_id": scene_id,
            "question": str(row["question"]),
            "answers": list(row.get("answers", [])),
            "image_path": str(image_path),
        })
    if scene_mismatches:
        raise ValueError(f"Image path/annotation scene mismatch: {scene_mismatches[:3]}")
    items.sort(key=lambda row: (row["scene_id"], number_suffix(row["question_id"])))
    total_matching = len(items)
    if max_questions is not None:
        items = items[:max_questions]
    if not items:
        raise RuntimeError("No shard overview images matched the ScanQA annotation and scene filter")
    return items, {
        "annotation_rows": len(annotations),
        "overview_images": len(overview_by_qid),
        "matching_questions": total_matching,
        "selected_questions": len(items),
        "unknown_image_question_ids": len(unknown_images),
        "unknown_image_examples": unknown_images[:10],
        "scenes": len({row["scene_id"] for row in items}),
    }


def question_rule(question: str) -> str:
    lowered = question.strip().lower()
    if lowered.startswith(("what color", "what is the color", "what colour")):
        return "Return one color word or short color phrase only; read color from the object mesh, not its label."
    if lowered.startswith(("how many", "how much")):
        return "Count distinct visible relevant meshes once and return one number only."
    return "Return one canonical short natural-language answer only."


def messages_for(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "url": str(Path(item["image_path"]).resolve())},
            {"type": "text", "text": f"Question: {item['question']}\n{question_rule(item['question'])}"},
        ]},
    ]


CODE_FENCE_PATTERN = re.compile(r"^```(?:python|text)?\s*|\s*```$", flags=re.IGNORECASE)


def normalize_prediction(raw_text: str, question: str) -> str:
    text = CODE_FENCE_PATTERN.sub("", str(raw_text).strip()).strip()
    text = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    try:
        parsed = ast.literal_eval(text)
        values = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        text = str(next((value for value in values if str(value).strip()), ""))
    except (SyntaxError, ValueError):
        pass
    if question.strip().lower().startswith(("how many", "how much")):
        number = re.search(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", text)
        if number:
            text = number.group(0)
    text = re.sub(r"^(?:the\s+answer\s+is|it\s+is)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\r\n.!,;:\"'")


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines):
                print(f"Ignore incomplete final JSONL line in {path}", file=sys.stderr, flush=True)
                continue
            raise
        rows[str(row["question_id"])] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def validate_resume(rows: dict[str, dict[str, Any]], args: argparse.Namespace, tag: str) -> None:
    incompatible = [
        question_id for question_id, row in rows.items()
        if row.get("model_id") != args.model_id or row.get("shard_tag") != tag
    ]
    if incompatible:
        raise RuntimeError(
            f"Existing output belongs to another model/shard ({incompatible[:3]}). Use --overwrite or another --output-dir."
        )


def normalized_exact(value: str) -> str:
    value = re.sub(r"[^\w\s]", " ", str(value).lower())
    value = re.sub(r"\b(?:a|an|the)\b", " ", value)
    return " ".join(value.split())


def shard_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonempty = [row for row in rows if str(row.get("prediction", "")).strip()]
    exact = sum(
        normalized_exact(row["prediction"]) in {normalized_exact(answer) for answer in row["answers"]}
        for row in nonempty
    )
    return {
        "num_scored": len(nonempty),
        "empty_predictions": len(rows) - len(nonempty),
        "exact_match": exact / len(nonempty) if nonempty else 0.0,
        "exact_match_policy": "single_prediction_matches_any_gt_reference",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "schema_version", "shard_tag", "shard_index", "num_shards", "question_id", "scene_id",
        "question", "answers", "image_path", "prediction", "prediction_raw", "model_id",
        "quantization", "dtype", "latency_s", "generated_tokens", "tokens_per_s",
        "peak_vram_gib", "timestamp_utc",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["answers"] = json.dumps(row.get("answers", []), ensure_ascii=False)
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    validate_args(args)
    tag = shard_tag(args)
    args.work_dir = args.work_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    data_root = prepare_shard_data(args, tag)
    scope = select_shard_scope(data_root, tag)
    overview_by_qid = discover_overviews(scope)
    annotation_path = prepare_annotation(args, data_root)
    eval_items, dataset_stats = load_eval_items(
        annotation_path, overview_by_qid, args.scene_ids, args.max_questions,
    )
    if args.prepare_only:
        print(json.dumps({
            "status": "prepared", "shard": tag, "data_root": str(data_root),
            "scope": str(scope), "annotation": str(annotation_path),
            "dataset": dataset_stats,
        }, ensure_ascii=False, indent=2))
        return

    precision_tag = "nf4" if args.load_in_4bit else f"full_{args.dtype}"
    run_dir = args.output_dir / safe_slug(args.model_id) / precision_tag / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    raw_path = run_dir / "raw_predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    csv_path = run_dir / "predictions.csv"
    summary_path = run_dir / "summary.json"
    if args.overwrite:
        for path in (predictions_path, raw_path, errors_path, csv_path, summary_path):
            path.unlink(missing_ok=True)

    completed = read_jsonl(predictions_path)
    errors = read_jsonl(errors_path)
    validate_resume(completed, args, tag)
    selected_qids = {item["question_id"] for item in eval_items}
    completed = {qid: row for qid, row in completed.items() if qid in selected_qids}
    errors = {qid: row for qid, row in errors.items() if qid in selected_qids and qid not in completed}
    ordered_completed = [completed[item["question_id"]] for item in eval_items if item["question_id"] in completed]
    atomic_write_jsonl(predictions_path, ordered_completed)
    atomic_write_jsonl(raw_path, (
        {"schema_version": SCHEMA_VERSION, "shard_tag": tag, "question_id": row["question_id"],
         "scene_id": row["scene_id"], "prediction_raw": row["prediction_raw"],
         "prediction": row["prediction"], "model_id": row["model_id"]}
        for row in ordered_completed
    ))
    atomic_write_jsonl(errors_path, errors.values())

    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install CUDA torch, then: pip install 'transformers>=4.57.0' "
            "accelerate gdown pillow tqdm"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.dtype == "float16":
        compute_dtype = torch.float16
    elif args.dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs: dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "--load-in-4bit requires bitsandbytes: pip install bitsandbytes"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    device_name = torch.cuda.get_device_name(0)
    print(json.dumps({
        "shard": tag, "scope": str(scope), "annotation": str(annotation_path),
        "output": str(run_dir), "model_id": args.model_id, "gpu": device_name,
        "dtype": str(compute_dtype), "load_in_4bit": args.load_in_4bit,
        "dataset": dataset_stats, "resuming": len(completed),
    }, ensure_ascii=False, indent=2), flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).eval()
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_started
    print(f"Model loaded in {model_load_s:.1f}s; VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    @torch.inference_mode()
    def predict(item: dict[str, Any]) -> tuple[str, str, float, int]:
        inputs = processor.apply_chat_template(
            messages_for(item), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        token_count = int(generated.shape[1] - inputs.input_ids.shape[1])
        raw = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        prediction = normalize_prediction(raw, item["question"])
        del inputs, generated
        return prediction, raw, elapsed, token_count

    pending = [item for item in eval_items if item["question_id"] not in completed]
    benchmark_started = time.perf_counter()
    interrupted = False
    try:
        for step, item in enumerate(pending, 1):
            try:
                prediction, raw, elapsed, token_count = predict(item)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "shard_tag": tag,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    **item,
                    "prediction": prediction,
                    "prediction_raw": raw,
                    "model_id": args.model_id,
                    "prompt_version": PROMPT_VERSION,
                    "quantization": "nf4_4bit" if args.load_in_4bit else "none",
                    "dtype": str(compute_dtype).replace("torch.", ""),
                    "latency_s": elapsed,
                    "generated_tokens": token_count,
                    "tokens_per_s": token_count / elapsed if elapsed else 0.0,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "timestamp_utc": utc_now(),
                }
                completed[item["question_id"]] = row
                errors.pop(item["question_id"], None)
                append_jsonl(predictions_path, row)
                append_jsonl(raw_path, {
                    "schema_version": SCHEMA_VERSION, "shard_tag": tag,
                    "question_id": item["question_id"], "scene_id": item["scene_id"],
                    "prediction_raw": raw, "prediction": prediction, "model_id": args.model_id,
                })
                print(
                    f"[{step}/{len(pending)}] {item['question_id']}: {elapsed:.2f}s, "
                    f"{token_count} tok, raw={raw!r}, normalized={prediction!r}", flush=True,
                )
            except Exception as exc:
                error = {
                    "schema_version": SCHEMA_VERSION, "shard_tag": tag,
                    "question_id": item["question_id"], "scene_id": item["scene_id"],
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(limit=8), "timestamp_utc": utc_now(),
                }
                errors[item["question_id"]] = error
                append_jsonl(errors_path, error)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[ERROR] {item['question_id']}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user; completed rows were already saved.", file=sys.stderr, flush=True)

    benchmark_wall_s = time.perf_counter() - benchmark_started
    ordered_rows = [completed[item["question_id"]] for item in eval_items if item["question_id"] in completed]
    remaining = [item["question_id"] for item in eval_items if item["question_id"] not in completed]
    atomic_write_jsonl(predictions_path, ordered_rows)
    atomic_write_jsonl(raw_path, (
        {"schema_version": SCHEMA_VERSION, "shard_tag": tag, "question_id": row["question_id"],
         "scene_id": row["scene_id"], "prediction_raw": row["prediction_raw"],
         "prediction": row["prediction"], "model_id": row["model_id"]}
        for row in ordered_rows
    ))
    atomic_write_jsonl(errors_path, (errors[qid] for qid in remaining if qid in errors))
    write_csv(csv_path, ordered_rows)
    generation_s = sum(float(row["latency_s"]) for row in ordered_rows)
    generated_tokens = sum(int(row["generated_tokens"]) for row in ordered_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "interrupted" if interrupted else ("complete" if not remaining else "partial_with_errors"),
        "timestamp_utc": utc_now(), "shard_tag": tag,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "model_id": args.model_id, "prompt_version": PROMPT_VERSION,
        "split": args.split, "annotation": str(annotation_path), "scope": str(scope),
        "gpu": device_name, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": str(compute_dtype).replace("torch.", ""),
        "quantization": "nf4_4bit" if args.load_in_4bit else "none",
        "attn_implementation": args.attn_implementation,
        "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens, "model_load_s": model_load_s,
        "benchmark_wall_s_this_session": benchmark_wall_s,
        "completed": len(ordered_rows), "remaining": len(remaining),
        "errors_for_remaining": sum(qid in errors for qid in remaining),
        "total_generation_s_across_saved_rows": generation_s,
        "total_generated_tokens_across_saved_rows": generated_tokens,
        "generation_tokens_per_s": generated_tokens / generation_s if generation_s else 0.0,
        "peak_vram_gib_this_session": torch.cuda.max_memory_allocated() / 2**30,
        "dataset": dataset_stats, "metrics": shard_metrics(ordered_rows),
        "artifacts": {
            "predictions_jsonl": str(predictions_path), "raw_predictions_jsonl": str(raw_path),
            "predictions_csv": str(csv_path), "errors_jsonl": str(errors_path),
        },
    }
    atomic_write_json(summary_path, summary)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nMerge-ready: {predictions_path}\nRaw output:  {raw_path}\nSummary:     {summary_path}", flush=True)


if __name__ == "__main__":
    main()
