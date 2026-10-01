# Run from PowerShell on the Windows machine that contains the RTX 3090.
# Leave $SceneId empty when the archive has exactly one scene. Set it only for
# an archive containing more than one scene.
$SceneUrl = 'https://drive.google.com/file/d/14Cc6PB9zewuIxg9B7SvublQuD1WomTUd/view?usp=sharing'
$SceneId = ''

python -m pip install -r benchmarks/requirements-qwen3vl-8b.txt
nvidia-smi

$SceneArgs = @()
if ($SceneId) { $SceneArgs = @('--scene-id', $SceneId) }

python benchmarks/benchmark_qwen3vl_8b_scene.py `
  --scene-url $SceneUrl `
  @SceneArgs `
  --split val `
  --dtype float16 `
  --attn-implementation sdpa `
  --max-images 0 `
  --max-new-tokens 32 `
  --work-dir benchmark_work `
  --output-dir outputs/qwen3vl8b_scene_benchmark
