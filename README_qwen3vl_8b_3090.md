# Benchmark Qwen3-VL-8B trên một scene (RTX 3090)

`benchmark_qwen3vl_8b_scene.py` đánh giá một scene ScanQA với `Qwen/Qwen3-VL-8B-Instruct` FP16, là cấu hình đầy đủ phù hợp với VRAM 24 GB của RTX 3090. Mặc định mỗi câu hỏi nhận **toàn bộ** render natural-color sau khi loại render có cùng tập object IDs. Script chạy tuần tự, lưu kết quả sau từng câu hỏi, nên có thể chạy lại an toàn sau khi mất kết nối.

## Chuẩn bị trên máy Linux có RTX 3090

Tạo môi trường Python 3.10+ và cài PyTorch CUDA phù hợp với driver trước. Ví dụ CUDA 12.4:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Nếu archive chỉ có một scene, runner tự nhận diện scene đó. Nếu archive có nhiều scene, đặt `SCENE_ID` trong `benchmarks/run_qwen3vl_8b_3090.sh` cho đúng scene cần chạy, rồi chạy:

```bash
chmod +x benchmarks/run_qwen3vl_8b_3090.sh
./benchmarks/run_qwen3vl_8b_3090.sh
```

Nếu chạy Windows native (PowerShell), đặt `$SceneId` khi archive có nhiều scene rồi dùng:

```powershell
.\benchmarks\run_qwen3vl_8b_3090.ps1
```

Script tự tải link Drive bằng `gdown`, giải nén ZIP vào `benchmark_work/extracted`, tải model từ Hugging Face ở lần chạy đầu và lưu artifacts vào `outputs/qwen3vl8b_scene_benchmark/`.

## Điều kiện dữ liệu

Archive cần chứa (trực tiếp hoặc bên trong thư mục `prepare-data`) cả hai phần:

```text
data/scanqa/ScanQA_v1.0_val.json
outputs/capruner_source_natural_colors/shard_*/manifest.csv
```

Các file PNG được manifest trỏ đến cũng phải nằm trong archive. Nếu link chỉ chứa raw ScanNet mesh, cần render bằng pipeline CAPruner/PoseAlign trước; raw mesh một mình chưa thể đưa trực tiếp vào Qwen-VL benchmark này.

## Lệnh trực tiếp và smoke test

Khi đã giải nén dữ liệu, không cần tải lại Drive:

```bash
python benchmarks/benchmark_qwen3vl_8b_scene.py \
  --data-root /path/to/prepare-data \
  --scene-id scene0030_00 \
  --max-questions 3 \
  --output-dir outputs/qwen3vl8b_scene_benchmark
```

Xóa `--max-questions 3` để benchmark toàn bộ câu hỏi scene. `--max-images 0` (mặc định) gửi toàn bộ ảnh sau dedup; đặt ví dụ `--max-images 5` để giới hạn 5 ảnh. `--overwrite` xóa log dự đoán/lỗi của đúng run tag; không có cờ này script tự resume.

Các file chính: `*_predictions.csv`, `*_errors.jsonl`, `*_report.json`. Mọi output của prompt là Python-style array các string dùng nháy đơn, giống GT: một đáp án là `['chair']`, nhiều đáp án là `['chair', 'table']`. Câu đếm chỉ có số: `['4']` hoặc `['2', '1']` khi có nhiều số cần trả theo thứ tự câu hỏi. Report bao gồm CIDEr, BLEU-4, METEOR, ROUGE-L, exact-match, latency generation, tokens/s và peak VRAM. METEOR yêu cầu Java; nếu máy không có Java, thêm `--skip-coco-metrics` để vẫn chạy inference và exact-match.

`sdpa` là lựa chọn ổn định mặc định. Nếu đã cài FlashAttention 2 đúng phiên bản CUDA/PyTorch, thay bằng `--attn-implementation flash_attention_2` để tiết kiệm VRAM/tăng tốc với nhiều ảnh.
