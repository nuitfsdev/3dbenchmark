# Hướng dẫn chạy benchmark Qwen-VL

Tài liệu này dùng cho terminal Linux trên máy NVIDIA, VPS, RunPod hoặc terminal của
Google Colab. Các lệnh được chạy từ thư mục project, ví dụ `/root/3dbenchmark`.

## 1. Chọn đúng runner

Repo có hai loại dữ liệu khác nhau:

| Dữ liệu | Dấu hiệu nhận biết | Runner |
|---|---|---|
| Natural-color, nhiều ảnh cho mỗi câu hỏi | Có `manifest.csv` và các file `source_rank_*.png` | `benchmark_qwen3vl_8b_scene.py` |
| Instance-mesh overview, một ảnh cho mỗi câu hỏi | Có `<question_id>/overview.png` | `benchmark_qwen3vl_8b_overview_shard.py` |

Tên `benchmark_qwen3vl_8b_scene.py` được giữ để tương thích, nhưng runner natural-color
đã dùng `AutoModelForImageTextToText`. Nó hỗ trợ cả Qwen2.5-VL và Qwen3-VL bằng
`--model-id`, đồng thời chạy được một scene hoặc toàn bộ một shard.

Không dùng runner overview cho ZIP natural-color. Runner overview luôn tìm file có tên
chính xác `overview.png`.

## 2. Chuẩn bị môi trường

Kiểm tra GPU:

```bash
nvidia-smi
python3 --version
```

Khuyến nghị Python 3.10 trở lên. Tạo môi trường riêng:

```bash
cd /root/3dbenchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Cài PyTorch CUDA. Đây là ví dụ wheel CUDA 12.4; nếu image GPU đã có PyTorch CUDA
thì không cần cài lại:

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124
```

Cài dependency benchmark:

```bash
python -m pip install -r benchmarks/requirements-qwen3vl-8b.txt
```

Nếu các script nằm ngay ở project root, đường dẫn requirements vẫn có thể là
`benchmarks/requirements-qwen3vl-8b.txt`.

Kiểm tra PyTorch nhìn thấy GPU:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

`available` phải là `True`.

## 3. Các file cần đặt trên máy GPU

Với benchmark natural-color:

```text
/root/3dbenchmark/
├── benchmark_qwen3vl_8b_scene.py
├── download_qwen_benchmark_data.py
├── benchmarks/
│   └── requirements-qwen3vl-8b.txt
├── downloads/
└── outputs/
```

Nếu giữ script trong `benchmarks/`, thêm tiền tố `benchmarks/` khi gọi Python.

## 4. Layout dữ liệu natural-color

### Cách khuyến nghị: ZIP chỉ chứa shard

Không cần nén `prepare-data`. Mỗi ZIP chỉ chứa một shard:

```text
shard_01_of_10.zip
└── shard_01_of_10/
    ├── manifest.csv
    ├── scene0011_00/
    │   ├── val-scene0011-0/
    │   │   ├── source_rank_01_node_003.png
    │   │   └── ...
    │   └── ...
    ├── scene0081_00/
    └── ...
```

Annotation được giữ riêng và dùng chung cho mọi shard:

```text
ScanQA_v1.0_val.json
```

### Layout bundle cũng được hỗ trợ

Runner vẫn nhận archive có cấu trúc đầy đủ:

```text
prepare-data/
├── data/scanqa/ScanQA_v1.0_val.json
└── outputs/capruner_source_natural_colors/
    └── shard_01_of_10/
        ├── manifest.csv
        └── ...
```

Với layout bundle, có thể bỏ `--annotation-file` vì runner tự tìm annotation.

## 5. Tải shard và annotation trước khi benchmark

Downloader chỉ tải và kiểm tra dữ liệu; nó không load model.

### Tải cả shard ZIP và ScanQA JSON từ Google Drive

```bash
python download_qwen_benchmark_data.py \
  --shard-index 1 \
  --num-shards 10 \
  --shard-url "LINK_DRIVE_SHARD_01_ZIP" \
  --annotation-url "LINK_DRIVE_SCANQA_VAL_JSON" \
  --split val \
  --output-dir downloads
```

Kết quả:

```text
downloads/
├── shard_01_of_10.zip
└── ScanQA_v1.0_val.json
```

### Annotation đã có sẵn trên máy

```bash
python download_qwen_benchmark_data.py \
  --shard-index 1 \
  --num-shards 10 \
  --shard-url "LINK_DRIVE_SHARD_01_ZIP" \
  --annotation-file /duong-dan/ScanQA_v1.0_val.json \
  --split val \
  --output-dir downloads
```

Downloader sẽ tái sử dụng file đã tải. Muốn tải lại:

```bash
python download_qwen_benchmark_data.py \
  --shard-index 1 \
  --num-shards 10 \
  --shard-url "LINK_DRIVE_SHARD_01_ZIP" \
  --annotation-url "LINK_DRIVE_SCANQA_VAL_JSON" \
  --output-dir downloads \
  --redownload
```

## 6. Natural-color: chạy toàn bộ một shard

### Smoke test Qwen2.5-VL-3B

Lệnh này load model một lần, chọn ba câu đầu của toàn shard và gửi tối đa năm ảnh
sau dedup cho mỗi câu:

```bash
python benchmark_qwen3vl_8b_scene.py \
  --scene-archive downloads/shard_01_of_10.zip \
  --annotation-file downloads/ScanQA_v1.0_val.json \
  --shard-index 1 \
  --num-shards 10 \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --split val \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-images 5 \
  --max-questions 3 \
  --max-new-tokens 32 \
  --skip-coco-metrics \
  --work-dir benchmark_work/qwen25vl3b_shard01 \
  --output-dir outputs/qwen25vl3b_natural
```

### Full run Qwen2.5-VL-3B

Bỏ `--max-questions`; `--max-images 0` dùng toàn bộ ảnh sau dedup:

```bash
python benchmark_qwen3vl_8b_scene.py \
  --scene-archive downloads/shard_01_of_10.zip \
  --annotation-file downloads/ScanQA_v1.0_val.json \
  --shard-index 1 \
  --num-shards 10 \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --split val \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-images 0 \
  --max-new-tokens 32 \
  --work-dir benchmark_work/qwen25vl3b_shard01 \
  --output-dir outputs/qwen25vl3b_natural
```

Qwen2.5-VL-3B cũng hỗ trợ NF4. Cài `bitsandbytes` rồi thêm:

```text
--load-in-4bit
```

### Full run Qwen3-VL-8B

```bash
python benchmark_qwen3vl_8b_scene.py \
  --scene-archive downloads/shard_01_of_10.zip \
  --annotation-file downloads/ScanQA_v1.0_val.json \
  --shard-index 1 \
  --num-shards 10 \
  --model-id "Qwen/Qwen3-VL-8B-Instruct" \
  --split val \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-images 0 \
  --max-new-tokens 32 \
  --work-dir benchmark_work/qwen3vl8b_shard01 \
  --output-dir outputs/qwen3vl8b_natural
```

### Qwen3-VL-8B NF4 4-bit

Cài bitsandbytes một lần:

```bash
python -m pip install bitsandbytes
```

Sau đó thêm `--load-in-4bit`. NF4 áp dụng cho cả chạy toàn shard và chạy một scene:

```bash
python benchmark_qwen3vl_8b_scene.py \
  --scene-archive downloads/shard_01_of_10.zip \
  --annotation-file downloads/ScanQA_v1.0_val.json \
  --shard-index 1 \
  --num-shards 10 \
  --model-id "Qwen/Qwen3-VL-8B-Instruct" \
  --load-in-4bit \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-images 0 \
  --max-new-tokens 32 \
  --work-dir benchmark_work/qwen3vl8b_nf4_shard01 \
  --output-dir outputs/qwen3vl8b_nf4_natural
```

Khi không truyền `--skip-coco-metrics`, natural-color report tính:

```text
exact_match_strict
exact_match_normalized
BLEU-1, BLEU-2, BLEU-3, BLEU-4
METEOR
ROUGE-L
CIDEr
SPICE
```

`exact_match` được giữ làm alias của `exact_match_normalized` để tương thích với report
cũ. METEOR và SPICE cần Java:

```bash
java -version
```

EM@10 không được tính vì runner sinh một đáp án cho mỗi câu, không xếp hạng mười đáp
án như classifier ScanQA gốc.

## 7. Natural-color: chạy một scene trong shard

Dùng cùng ZIP và annotation. Thay cặp `--shard-index/--num-shards` bằng `--scene-id`:

```bash
python benchmark_qwen3vl_8b_scene.py \
  --scene-archive downloads/shard_01_of_10.zip \
  --annotation-file downloads/ScanQA_v1.0_val.json \
  --scene-id scene0011_00 \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --split val \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-images 5 \
  --max-questions 3 \
  --skip-coco-metrics \
  --work-dir benchmark_work/qwen25vl3b_shard01 \
  --output-dir outputs/qwen25vl3b_natural
```

Không truyền `--scene-id` cùng `--shard-index`.

## 8. Natural-color: chạy trực tiếp từ Google Drive

Cách này tiện cho lần đầu, nhưng chạy lại có thể phụ thuộc kết nối. Cách tải riêng ở
mục 5 ổn định hơn.

```bash
python benchmark_qwen3vl_8b_scene.py \
  --scene-url "LINK_DRIVE_SHARD_01_ZIP" \
  --annotation-url "LINK_DRIVE_SCANQA_VAL_JSON" \
  --shard-index 1 \
  --num-shards 10 \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --split val \
  --dtype float16 \
  --max-images 5 \
  --max-questions 3 \
  --skip-coco-metrics \
  --work-dir benchmark_work/direct_drive \
  --output-dir outputs/qwen25vl3b_natural
```

## 9. Natural-color: dữ liệu đã giải nén

Nếu `shard_01_of_10/` đã có trên máy:

```bash
python benchmark_qwen3vl_8b_scene.py \
  --data-root /duong-dan/chua-cac-folder-shard \
  --annotation-file /duong-dan/ScanQA_v1.0_val.json \
  --shard-index 1 \
  --num-shards 10 \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --max-images 5 \
  --max-questions 3 \
  --skip-coco-metrics \
  --output-dir outputs/qwen25vl3b_natural
```

`--data-root` có thể trỏ trực tiếp đến folder cha của `shard_01_of_10`; runner tự tìm
`shard_*/manifest.csv`.

## 10. Overview: layout và lệnh chạy

Chỉ dùng mục này khi mỗi question có đúng một `overview.png`:

```text
shard_01_of_10.zip
└── shard_01_of_10/
    └── scene0011_00/
        └── val-scene0011-0/
            └── overview.png
```

Smoke test Qwen2.5-VL-3B:

```bash
python benchmark_qwen3vl_8b_overview_shard.py \
  --shard-index 1 \
  --num-shards 10 \
  --shard-archive downloads/shard_01_of_10.zip \
  --annotation-file downloads/ScanQA_v1.0_val.json \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --split val \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-questions 3 \
  --work-dir benchmark_work/overview \
  --output-dir outputs/overview
```

Kiểm tra dữ liệu mà chưa load model:

```bash
python benchmark_qwen3vl_8b_overview_shard.py \
  --shard-index 1 \
  --num-shards 10 \
  --shard-archive downloads/shard_01_of_10.zip \
  --annotation-file downloads/ScanQA_v1.0_val.json \
  --prepare-only
```

Overview runner hỗ trợ 4-bit NF4:

```bash
python -m pip install bitsandbytes
```

Thêm cờ sau vào lệnh overview:

```text
--load-in-4bit
```

Natural-color và overview runner đều hỗ trợ NF4 bằng `--load-in-4bit`.

## 11. Resume, chạy lại và output

Hai runner ghi kết quả sau từng câu hỏi. Nếu tiến trình bị dừng, chạy lại đúng command
và cùng `--output-dir` để tiếp tục.

Natural-color tạo các file:

```text
*_predictions.jsonl
*_predictions.csv
*_errors.jsonl
*_report.json
```

Overview tạo:

```text
predictions.jsonl
raw_predictions.jsonl
predictions.csv
errors.jsonl
summary.json
```

Muốn bỏ kết quả cũ và chạy lại từ đầu, thêm:

```text
--overwrite
```

Nếu thay ZIP nhưng dùng lại cùng extraction folder, chọn một `--work-dir` mới. Overview
runner còn hỗ trợ `--overwrite-extract`; natural-color runner tái sử dụng extraction đã
hoàn tất.

## 12. Ý nghĩa các tham số thường dùng

| Tham số | Ý nghĩa |
|---|---|
| `--shard-index 1 --num-shards 10` | Chạy mọi scene/question trong shard 1 |
| `--scene-id scene0011_00` | Chỉ chạy một scene |
| `--max-questions 3` | Smoke test ba câu |
| `--max-images 5` | Tối đa năm ảnh sau dedup cho mỗi câu |
| `--max-images 0` | Dùng mọi ảnh sau dedup |
| `--no-deduplicate` | Giữ cả ảnh có cùng tập object IDs |
| `--dtype float16` | Phù hợp NVIDIA T4/RTX 3090 |
| `--dtype bfloat16` | Dùng khi GPU hỗ trợ BF16 |
| `--load-in-4bit` | Nạp trọng số bằng bitsandbytes NF4 |
| `--attn-implementation sdpa` | Không cần cài FlashAttention |
| `--skip-coco-metrics` | Không chạy CIDEr/BLEU/METEOR/ROUGE/SPICE; vẫn có inference và hai loại exact match |

## 13. Lỗi thường gặp

### `can't open file ... No such file or directory`

Kiểm tra vị trí script:

```bash
pwd
find . -maxdepth 3 -type f -name 'benchmark*.py'
```

### `gdown: unrecognized arguments: --fuzzy`

Các script mới không còn truyền `--fuzzy`. Upload lại script mới hoặc tải thủ công:

```bash
python -m gdown "LINK_DRIVE" -O downloads/shard_01_of_10.zip
```

### `No <question_id>/overview.png found`

Bạn đang dùng runner overview cho dữ liệu natural-color. Chuyển sang
`benchmark_qwen3vl_8b_scene.py`.

### `Không tìm thấy shard_*/manifest.csv`

Kiểm tra ZIP:

```bash
unzip -l downloads/shard_01_of_10.zip | head -50
unzip -l downloads/shard_01_of_10.zip | grep 'manifest.csv'
```

### `Không tìm thấy annotation`

Truyền một trong hai:

```text
--annotation-file downloads/ScanQA_v1.0_val.json
--annotation-url "LINK_DRIVE_SCANQA_VAL_JSON"
```

### `CUDA GPU is required` hoặc `torch.cuda.is_available() == False`

PyTorch đang là bản CPU hoặc NVIDIA driver chưa hoạt động. Kiểm tra lại `nvidia-smi`
và cài PyTorch CUDA.

### CUDA out of memory

Giảm số ảnh:

```text
--max-images 3
```

Hoặc thử model 3B trước:

```text
--model-id Qwen/Qwen2.5-VL-3B-Instruct
```

Với cả natural-color và overview runner có thể thêm `--load-in-4bit`.

### Đã thay ZIP nhưng runner vẫn dùng dữ liệu cũ

Dùng `--work-dir` mới, ví dụ:

```text
--work-dir benchmark_work/shard01_v2
```

## 14. Quy trình khuyến nghị

1. Cài dependency và xác nhận CUDA hoạt động.
2. Dùng `download_qwen_benchmark_data.py` tải ZIP cùng annotation.
3. Chạy smoke test với Qwen2.5-VL-3B, `--max-questions 3 --max-images 5`.
4. Kiểm tra prediction và error log.
5. Bỏ `--max-questions`, chọn `--max-images 0` để chạy toàn shard.
6. Chạy lại cùng command nếu tiến trình bị ngắt.
