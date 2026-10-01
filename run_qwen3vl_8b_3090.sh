#!/usr/bin/env bash
set -euo pipefail

# Run on the Linux machine that contains the RTX 3090.
# Leave SCENE_ID empty when the archive has exactly one scene; the Python script
# will detect it.  Set it explicitly only for an archive containing many scenes.
SCENE_URL="https://drive.google.com/file/d/14Cc6PB9zewuIxg9B7SvublQuD1WomTUd/view?usp=sharing"
SCENE_ID=""

python -m pip install -r benchmarks/requirements-qwen3vl-8b.txt
nvidia-smi

scene_args=()
if [[ -n "$SCENE_ID" ]]; then scene_args=(--scene-id "$SCENE_ID"); fi

python benchmarks/benchmark_qwen3vl_8b_scene.py \
  --scene-url "$SCENE_URL" \
  "${scene_args[@]}" \
  --split val \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-images 0 \
  --max-new-tokens 32 \
  --work-dir benchmark_work \
  --output-dir outputs/qwen3vl8b_scene_benchmark
