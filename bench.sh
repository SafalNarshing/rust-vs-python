#!/usr/bin/env bash
# Runs the Python and Rust benchmarks one after the other while logging GPU
# metrics with nvidia-smi and process metrics with /usr/bin/time.
# Usage: ./bench.sh            (run both)
#        ./bench.sh python     (only one side)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL="$ROOT/qwen2.5-1.5b"
OUT="$ROOT/results"
COOLDOWN=30   # seconds between runs so the GPU starts each run at a similar temperature
mkdir -p "$OUT"

[[ -f "$MODEL/model.safetensors" ]] || { echo "Model not found in $MODEL"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
[[ -x /usr/bin/time ]] || { echo "Install GNU time: sudo pacman -S time"; exit 1; }

run() {
  local name=$1; shift
  echo; echo "=== $name: cooling down ${COOLDOWN}s ==="
  sleep "$COOLDOWN"

  # GPU memory in use before the run (desktop/compositor usage), subtracted later
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 > "$OUT/baseline_$name.txt"

  # Whole-GPU telemetry every 200 ms: utilization, VRAM, power, SM clock, temperature
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 200 > "$OUT/gpu_$name.csv" &
  local smi_pid=$!
  # Per-process VRAM (for reference; may be empty on some laptop driver setups)
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits -lms 200 > "$OUT/procs_$name.csv" &
  local procs_pid=$!
  trap 'kill $smi_pid $procs_pid 2>/dev/null || true' EXIT

  echo "=== $name: running ==="
  /usr/bin/time -v -o "$OUT/time_$name.txt" "$@"

  kill "$smi_pid" "$procs_pid" 2>/dev/null || true
  trap - EXIT
}

WHICH="${1:-both}"
if [[ "$WHICH" == "both" || "$WHICH" == "python" ]]; then
  run python "$ROOT/python/.venv/bin/python" "$ROOT/python/bench.py" "$MODEL" "$OUT/python.json"
fi
if [[ "$WHICH" == "both" || "$WHICH" == "rust" ]]; then
  run rust "$ROOT/rust/target/release/qwen-bench" "$MODEL" "$OUT/rust.json"
fi

# Disk footprint of what each side needs to run (model weights excluded; they are shared)
{
  echo "python_venv_bytes $(du -sb "$ROOT/python/.venv" | cut -f1)"
  echo "rust_binary_bytes $(stat -c %s "$ROOT/rust/target/release/qwen-bench")"
  echo "cuda_toolkit_bytes $(du -sb /opt/cuda 2>/dev/null | cut -f1 || echo 0)"
} > "$OUT/sizes.txt"

echo
python3 "$ROOT/summarize.py" "$OUT"
