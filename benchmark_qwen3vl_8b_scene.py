#!/usr/bin/env python3
"""Evaluate a supported vision-language model on natural-color ScanQA renders.

The input directory must contain the ScanQA annotations and CAPruner natural-colour
renders, for example::

    prepare-data/
      data/scanqa/ScanQA_v1.0_val.json
      outputs/capruner_source_natural_colors/shard_*/manifest.csv

The same data layout supports either one scene (``--scene-id``) or one complete
shard (``--shard-index`` plus ``--num-shards``).  Archives may contain the
``prepare-data`` tree above or place the annotation and shard directory at any
common root; the runner discovers both recursively.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any


MODEL_ID_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT_VERSION = "scanqa_posealign_flu_v11_count_numbers_only"
SYSTEM_PROMPT = """Answer the 3D question using all partial renders of the same scene. They share one pose;
each render shows a different object subset. Use IDs only to merge repeated objects across images: never
output IDs, labels, or parentheses, and count each ID once.

Coordinates are FLU: +X forward, +Y left, +Z up. Screen-left is +Y and screen-right is -Y; answer
left/right from object geometry, not label placement. For color, use the filled polygon, never its label.

Return every distinct supported answer, once; do not choose an arbitrary visible result. ALWAYS output a
Python-style array of short single-quoted strings, including one answer: `['chair']`. For counts, return
only the requested numbers as strings: `['4']` for one quantity or `['2', '1']` for multiple scopes in
question order. Never include object names or labels in a count answer. Output only the array, with no
explanation or sentence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_argument_group("scene input")
    source.add_argument("--data-root", type=Path, help="Existing prepare-data directory.")
    source.add_argument("--scene-archive", type=Path, help="ZIP archive downloaded from the supplied Drive link.")
    source.add_argument("--scene-url", help="Google Drive sharing URL; downloaded with gdown when --scene-archive is omitted.")
    source.add_argument("--work-dir", type=Path, default=Path("benchmark_work"), help="Download/extraction directory.")
    parser.add_argument("--annotation-file", type=Path,
                        help="Local ScanQA JSON when the shard ZIP contains images only.")
    parser.add_argument("--annotation-url",
                        help="Google Drive URL of a ScanQA JSON when the shard ZIP contains images only.")
    parser.add_argument("--scene-id", help="ScanNet scene, e.g. scene0030_00. Omit only when the archive has one scene.")
    parser.add_argument("--shard-index", type=int,
                        help="Run every scene/question in this one-based natural-color shard.")
    parser.add_argument("--num-shards", type=int,
                        help="Total shard count; required together with --shard-index.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test_w_obj", "test_wo_obj"))
    parser.add_argument("--image-root", type=Path, help="Override CAPruner render root under data-root.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3vl8b_scene_benchmark"))
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16",
                        help="FP16 is the recommended full-precision setting for a 24-GB RTX 3090.")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False,
                        help="Load model weights with bitsandbytes NF4 quantization.")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Top source-rank images after deduplication; 0 means send every available image.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-questions", type=int, help="Use a small value for a smoke test.")
    parser.add_argument("--no-deduplicate", action="store_true", help="Keep repeated object-set renders.")
    parser.add_argument("--overwrite", action="store_true", help="Discard prior prediction/error logs for this run name.")
    parser.add_argument("--skip-coco-metrics", action="store_true",
                        help="Write predictions and exact-match only; skip BLEU/METEOR/ROUGE/CIDEr/SPICE.")
    return parser.parse_args()


def run_or_raise(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def normalize_drive_url(url: str) -> str:
    match = re.search(r"drive\.google\.com/file/d/([^/?#]+)", url)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"
    return url


def extract_scene_archive(archive: Path, target: Path) -> Path:
    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Không tìm thấy scene archive: {archive}")
    marker = target / ".extract_complete"
    if not marker.is_file():
        target.mkdir(parents=True, exist_ok=True)
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        else:
            raise ValueError(f"Chỉ hỗ trợ .zip, nhận được: {archive.name}")
        marker.write_text(archive.name, encoding="utf-8")
    return target


def resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_root:
        candidates = [args.data_root.expanduser()]
    else:
        archive = args.scene_archive
        if archive is None and args.scene_url:
            download_dir = args.work_dir.expanduser() / "downloads"
            download_dir.mkdir(parents=True, exist_ok=True)
            archive = download_dir / "scene.zip"
            if not archive.is_file():
                run_or_raise([sys.executable, "-m", "gdown", normalize_drive_url(args.scene_url), "-O", str(archive)])
        if archive is None:
            raise ValueError("Cần một trong: --data-root, --scene-archive, hoặc --scene-url")
        extracted = extract_scene_archive(archive, args.work_dir.expanduser() / "extracted")
        candidates = [extracted, extracted / "prepare-data"] + [p for p in extracted.glob("**/prepare-data") if p.is_dir()]

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "data" / "scanqa").is_dir():
            return candidate
        manifests = list(candidate.rglob("shard_*/manifest.csv")) if candidate.is_dir() else []
        annotations = list(candidate.rglob("ScanQA_v1.0_*.json")) if candidate.is_dir() else []
        if manifests and (annotations or args.annotation_file or args.annotation_url):
            return candidate
    shown = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Không tìm thấy shard_*/manifest.csv. Annotation có thể nằm trong archive hoặc được truyền "
        "bằng --annotation-file/--annotation-url. Đã kiểm tra:\n  - " + shown
    )


def find_questions_file(data_root: Path, split: str, annotation_file: Path | None = None) -> Path:
    names = {
        "train": "ScanQA_v1.0_train.json",
        "val": "ScanQA_v1.0_val.json",
        "test_w_obj": "ScanQA_v1.0_test_w_obj.json",
        "test_wo_obj": "ScanQA_v1.0_test_wo_obj.json",
    }
    if annotation_file:
        path = annotation_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy annotation: {path}")
        return path
    path = data_root / "data" / "scanqa" / names[split]
    if path.is_file():
        return path
    matches = sorted(data_root.rglob(names[split]))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(f"Tìm thấy nhiều annotation {names[split]}: {matches[:5]}")
    raise FileNotFoundError(f"Không tìm thấy annotation {names[split]} dưới {data_root}")


def resolve_annotation(args: argparse.Namespace, data_root: Path) -> Path:
    if args.annotation_file:
        return find_questions_file(data_root, args.split, args.annotation_file)
    if args.annotation_url:
        names = {
            "train": "ScanQA_v1.0_train.json",
            "val": "ScanQA_v1.0_val.json",
            "test_w_obj": "ScanQA_v1.0_test_w_obj.json",
            "test_wo_obj": "ScanQA_v1.0_test_wo_obj.json",
        }
        destination = args.work_dir.expanduser() / "annotations" / names[args.split]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            run_or_raise([sys.executable, "-m", "gdown", normalize_drive_url(args.annotation_url), "-O", str(destination)])
        return destination.resolve()
    return find_questions_file(data_root, args.split)


def resolve_image_root(data_root: Path, override: Path | None) -> Path:
    if override:
        root = override.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Không tìm thấy image root: {root}")
        return root
    preferred = data_root / "outputs" / "capruner_source_natural_colors"
    if preferred.is_dir():
        return preferred.resolve()
    roots = sorted({
        manifest.parent.parent.resolve()
        for manifest in data_root.rglob("manifest.csv")
        if re.fullmatch(r"shard_\d+_of_\d+", manifest.parent.name)
    })
    named = [root for root in roots if root.name == "capruner_source_natural_colors"]
    if len(named) == 1:
        return named[0]
    if len(roots) == 1:
        return roots[0]
    if not roots:
        raise FileNotFoundError(f"Không tìm thấy shard_*/manifest.csv dưới {data_root}")
    raise ValueError(f"Có nhiều image roots; chỉ định --image-root. Ví dụ: {roots[:5]}")


def resolve_scene_id(image_root: Path, requested_scene_id: str | None) -> str:
    if requested_scene_id:
        return requested_scene_id
    scene_ids: set[str] = set()
    for manifest_path in image_root.glob("shard_*/manifest.csv"):
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            scene_ids.update(row["scene_id"] for row in csv.DictReader(handle) if row.get("scene_id"))
    if len(scene_ids) == 1:
        scene_id = next(iter(scene_ids))
        print(f"Tự nhận diện scene_id duy nhất trong archive: {scene_id}", flush=True)
        return scene_id
    shown = ", ".join(sorted(scene_ids)[:15])
    raise ValueError(f"Archive có {len(scene_ids)} scenes; phải chỉ định --scene-id. Ví dụ: {shown}")


def normalise_object_ids(value: Any) -> str:
    ids = [part.strip() for part in str(value).split(";") if part.strip()]
    try:
        return ";".join(map(str, sorted(map(int, ids))))
    except ValueError:
        return ";".join(sorted(ids))


def resolve_image_path(row: dict[str, str], manifest_dir: Path, data_root: Path) -> Path:
    raw = str(row.get("image_path", ""))
    filename = PureWindowsPath(raw).name
    candidates = (
        manifest_dir / str(row["scene_id"]) / str(row["question_id"]) / filename,
        data_root / raw.replace("\\", "/"),
        Path(raw),
    )
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0])


def number_suffix(question_id: str) -> int | str:
    match = re.search(r"-(\d+)$", question_id)
    return int(match.group(1)) if match else question_id


def load_eval_items(data_root: Path, image_root: Path, questions_path: Path, scene_id: str | None,
                    shard_tag: str | None, deduplicate: bool, max_images: int,
                    max_questions: int | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest_files = ([image_root / shard_tag / "manifest.csv"] if shard_tag
                      else sorted(image_root.glob("shard_*/manifest.csv")))
    manifest_files = [path for path in manifest_files if path.is_file()]
    if not manifest_files:
        raise FileNotFoundError(f"Không có shard_*/manifest.csv dưới {image_root}")
    rows: list[dict[str, str]] = []
    for manifest_path in manifest_files:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if scene_id is None or row.get("scene_id") == scene_id:
                    row["_manifest_dir"] = str(manifest_path.parent)
                    row["resolved_path"] = str(resolve_image_path(row, manifest_path.parent, data_root))
                    if Path(row["resolved_path"]).is_file():
                        rows.append(row)
    if not rows:
        scope = shard_tag or scene_id
        raise FileNotFoundError(f"Không có ảnh render tồn tại cho {scope} trong {image_root}")

    rows.sort(key=lambda row: (row["question_id"], int(float(row.get("source_rank") or 10**9))))
    raw_counts: dict[str, int] = {}
    image_paths: dict[str, list[str]] = {}
    seen_signatures: set[tuple[str, str]] = set()
    for row in rows:
        qid = row["question_id"]
        raw_counts[qid] = raw_counts.get(qid, 0) + 1
        signature = normalise_object_ids(row.get("object_ids", ""))
        if deduplicate and (qid, signature) in seen_signatures:
            continue
        seen_signatures.add((qid, signature))
        choices = image_paths.setdefault(qid, [])
        image = row["resolved_path"]
        if image not in choices and (max_images == 0 or len(choices) < max_images):
            choices.append(image)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if scene_id is not None:
        scene_questions = sorted(
            (q for q in questions if q.get("scene_id") == scene_id),
            key=lambda q: number_suffix(q["question_id"]),
        )
    else:
        scene_questions = sorted(
            (q for q in questions if q.get("question_id") in image_paths),
            key=lambda q: (str(q.get("scene_id", "")), number_suffix(q["question_id"])),
        )
    items_with_images = [
        {
            "question_id": q["question_id"], "scene_id": q["scene_id"], "question": q["question"],
            "answers": q.get("answers", []), "image_paths": image_paths[q["question_id"]],
            "num_manifest_images": raw_counts[q["question_id"]],
        }
        for q in scene_questions if q["question_id"] in image_paths
    ]
    items = items_with_images[:max_questions] if max_questions is not None else items_with_images
    if not items:
        raise RuntimeError(f"Không ghép được câu hỏi ScanQA nào có ảnh cho {scene_id}")
    stats = {
        "scenes": len({item["scene_id"] for item in items_with_images}),
        "scene_questions": len(scene_questions), "questions_with_images": len(items_with_images),
        "evaluated_questions": len(items),
        "manifest_images": len(rows), "images_sent": sum(len(item["image_paths"]) for item in items),
        "missing_questions": len(scene_questions) - len(items_with_images),
    }
    return items, stats


def answer_rule(question: str) -> str:
    question = question.strip().lower()
    if question.startswith(("what color", "what is the color", "what colour")):
        return "Required answer type: color words only."
    if question.startswith(("how many", "how much")):
        return "Required answer type: a single-quoted Python-style array containing only every requested number, in question order. Count unique IDs, not repeated views."
    if question.startswith(("where", "what side", "in what part")):
        return "Required answer type: short location phrase only."
    return "Required answer type: shortest complete semantic answer; include all valid distinct results, never an instance ID."


def messages_for(item: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "image", "url": path} for path in item["image_paths"]]
    content.append({"type": "text", "text": f"Question: {item['question']}\n{answer_rule(item['question'])}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {row["question_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def write_jsonl(rows: dict[str, dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_answer(value: str) -> str:
    text = re.sub(r"^(answer\s*:\s*)", "", str(value).strip(), flags=re.IGNORECASE).strip().strip("\"'")
    # Preserve the model's JSON-array prediction in the output file, but flatten it
    # for text metrics whose ScanQA references are plain captions.
    if text.startswith("["):
        try:
            values = ast.literal_eval(text)
            if isinstance(values, list) and all(isinstance(item, (str, int, float)) for item in values):
                return ", ".join(str(item) for item in values)
        except (SyntaxError, ValueError):
            pass
    return text


def score_predictions(rows: list[dict[str, Any]], skip_coco: bool) -> dict[str, Any]:
    valid = [row for row in rows if clean_answer(row["prediction"])]
    strict = sum(
        clean_answer(row["prediction"]) in {clean_answer(a) for a in row["answers"]}
        for row in valid
    )
    normalized = sum(
        clean_answer(row["prediction"]).lower()
        in {clean_answer(a).lower() for a in row["answers"]}
        for row in valid
    )
    denominator = len(valid)
    result: dict[str, Any] = {
        "exact_match": normalized / denominator if denominator else 0.0,
        "exact_match_strict": strict / denominator if denominator else 0.0,
        "exact_match_normalized": normalized / denominator if denominator else 0.0,
        "num_scored": denominator,
    }
    if skip_coco or not valid:
        return result
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        gts = {row["question_id"]: [{"caption": clean_answer(a)} for a in row["answers"]] for row in valid}
        res = {row["question_id"]: [{"caption": clean_answer(row["prediction"])}] for row in valid}
        tokenizer = PTBTokenizer()
        gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)
        for scorer, names in ((Cider(), ["CIDEr"]), (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
                              (Meteor(), ["METEOR"]), (Rouge(), ["ROUGE-L"]),
                              (Spice(), ["SPICE"])):
            scores, _ = scorer.compute_score(gts, res)
            if not isinstance(scores, (list, tuple)):
                scores = [scores]
            result.update({name: float(score) for name, score in zip(names, scores)})
    except Exception as exc:  # Exact match and all inference artifacts remain valid.
        result["coco_metrics_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    args = parse_args()
    if args.max_images < 0 or args.max_new_tokens < 1:
        raise ValueError("--max-images phải >= 0 và --max-new-tokens phải lớn hơn 0")
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard-index và --num-shards phải được truyền cùng nhau")
    if args.shard_index is not None and not 1 <= args.shard_index <= args.num_shards:
        raise ValueError("Cần 1 <= --shard-index <= --num-shards")
    if args.shard_index is not None and args.scene_id:
        raise ValueError("Chọn một trong --scene-id hoặc --shard-index, không dùng cả hai")
    if args.annotation_file and args.annotation_url:
        raise ValueError("Chọn một trong --annotation-file hoặc --annotation-url")
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit("Thiếu dependency. Cài torch CUDA trước, sau đó: pip install -r benchmarks/requirements-qwen3vl-8b.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy CUDA GPU. Script này cần NVIDIA GPU; RTX 3090 phải hiện trong nvidia-smi.")

    data_root = resolve_data_root(args)
    questions_path = resolve_annotation(args, data_root)
    image_root = resolve_image_root(data_root, args.image_root)
    if args.shard_index is not None:
        width = max(2, len(str(args.num_shards)))
        shard_tag = f"shard_{args.shard_index:0{width}d}_of_{args.num_shards:0{width}d}"
        scene_id = None
        scope_tag = shard_tag
    else:
        shard_tag = None
        scene_id = resolve_scene_id(image_root, args.scene_id)
        scope_tag = scene_id
    eval_items, dataset_stats = load_eval_items(
        data_root, image_root, questions_path, scene_id, shard_tag,
        not args.no_deduplicate, args.max_images, args.max_questions,
    )
    image_tag = "allimages" if args.max_images == 0 else f"top{args.max_images}"
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model_id).strip("-_").lower()
    precision_tag = "nf4" if args.load_in_4bit else args.dtype
    run_tag = f"{scope_tag}_{model_tag}_{precision_tag}_{image_tag}_{'all' if args.no_deduplicate else 'unique'}_promptv11"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path, error_path = output_dir / f"{run_tag}_predictions.jsonl", output_dir / f"{run_tag}_errors.jsonl"
    if args.overwrite:
        for path in (prediction_path, error_path):
            path.unlink(missing_ok=True)
    completed, failed = load_jsonl(prediction_path), load_jsonl(error_path)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device_name = torch.cuda.get_device_name(0)
    print(json.dumps({"data_root": str(data_root), "image_root": str(image_root), "run_tag": run_tag,
                      "gpu": device_name, "cuda": torch.version.cuda, "dtype": args.dtype,
                      "quantization": "nf4_4bit" if args.load_in_4bit else "none",
                      "dataset": dataset_stats, "resuming_predictions": len(completed), "resuming_errors": len(failed)}, indent=2), flush=True)
    if "3090" not in device_name.lower():
        print(f"WARNING: GPU is '{device_name}', not RTX 3090. Continuing as requested.", file=sys.stderr)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_id)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401 -- fail early with a useful message.
        except ImportError as exc:
            raise SystemExit(
                "--load-in-4bit cần bitsandbytes. Cài bằng: python -m pip install bitsandbytes"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs).eval()
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start
    print(f"Model loaded in {model_load_s:.1f}s; allocated VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    @torch.inference_mode()
    def predict(item: dict[str, Any]) -> tuple[str, float, int]:
        inputs = processor.apply_chat_template(messages_for(item), add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new_tokens = int(generated.shape[1] - inputs.input_ids.shape[1])
        text = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
        del inputs, generated
        return text, elapsed, new_tokens

    pending = [item for item in eval_items if item["question_id"] not in completed and item["question_id"] not in failed]
    all_start = time.perf_counter()
    for index, item in enumerate(pending, start=1):
        try:
            prediction, elapsed, new_tokens = predict(item)
            completed[item["question_id"]] = {**item, "prediction": prediction, "model_id": args.model_id,
                                                 "latency_s": elapsed, "generated_tokens": new_tokens,
                                                 "tokens_per_s": new_tokens / elapsed if elapsed else 0.0,
                                                 "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            print(f"[{index}/{len(pending)}] {item['question_id']}: {elapsed:.2f}s, {new_tokens} tok, {prediction!r}", flush=True)
        except Exception as exc:
            failed[item["question_id"]] = {"question_id": item["question_id"], "scene_id": item["scene_id"],
                                             "error_type": type(exc).__name__, "error": str(exc),
                                             "traceback": traceback.format_exc(limit=5),
                                             "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[SKIP] {item['question_id']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        write_jsonl(completed, prediction_path)
        write_jsonl(failed, error_path)
    total_inference_s = time.perf_counter() - all_start

    rows = [completed[item["question_id"]] for item in eval_items if item["question_id"] in completed]
    csv_path = output_dir / f"{run_tag}_predictions.csv"
    if rows:
        fields = list(rows[0])
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    metrics = score_predictions(rows, args.skip_coco_metrics)
    generated_tokens = sum(row.get("generated_tokens", 0) for row in rows)
    latency = sum(row.get("latency_s", 0.0) for row in rows)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model_id": args.model_id,
              "scene_id": scene_id, "shard_tag": shard_tag,
              "split": args.split, "data_root": str(data_root), "image_root": str(image_root), "gpu": device_name,
              "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": args.dtype,
              "quantization": "nf4_4bit" if args.load_in_4bit else "none",
              "attn_implementation": args.attn_implementation, "model_load_s": model_load_s,
              "benchmark_wall_s": total_inference_s, "total_generation_s": latency,
              "total_generated_tokens": generated_tokens, "generation_tokens_per_s": generated_tokens / latency if latency else 0.0,
              "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "dataset": dataset_stats,
              "completed": len(rows), "failed": len(failed), "metrics": metrics, "prompt_version": PROMPT_VERSION,
              "deduplicate_by_object_set": not args.no_deduplicate, "max_images": args.max_images,
              "max_new_tokens": args.max_new_tokens}
    report_path = output_dir / f"{run_tag}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDONE")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Predictions: {csv_path}\nErrors:      {error_path}\nReport:      {report_path}")


if __name__ == "__main__":
    main()
