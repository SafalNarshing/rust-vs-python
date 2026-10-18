# rust-vs-python

Both sides use identical settings: Qwen2.5-1.5B-Instruct, BF16, the same tokenizer and chat template, greedy decoding, and 20 prompts × 256 generated tokens each.

## Results

RTX 4070 Laptop GPU (8 GB, 55 W power limit), Intel Core Ultra 9 185H, driver 615.78, CUDA 13.4. PyTorch 2.11.0 vs Candle 0.11. Median of 3 runs.

| Metric | PyTorch (Python) | Candle (Rust) |
| --- | --- | --- |
| Decode speed (tokens/s) | 61.6 | 61.1 |
| Time to first token (ms) | 17.7 | 17.9 |
| Generation time, 20 prompts (s) | 83.0 | 83.7 |
| Startup to model ready (s) | 2.96 | 0.73 |
| Peak VRAM (MB) | 3,330 | 3,148 |
| Peak system RAM (MB) | 3,897 | 3,639 |
| Install size | 6,752 MB venv (CUDA libs included) | 17 MB binary + system CUDA |

Inside the model it's a tie. Around the model (startup, memory), Rust wins.

## Requirements

- An NVIDIA GPU (RTX 20 series or newer) with a working driver: `nvidia-smi` should list your GPU
- CUDA toolkit 12.x or 13.x, with `nvcc` on your `PATH`
- A C/C++ compiler: GCC on Linux, Visual Studio Build Tools (C++ workload) on Windows
- Rust (stable)
- [uv](https://github.com/astral-sh/uv)
- GNU `time` (Linux only, used by `bench.sh`)
- About 15 GB of free disk space

## Setup and run

### Linux

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/rust-vs-python-gpu.git
cd rust-vs-python-gpu
chmod +x bench.sh
```

**2. Set up Python**

```bash
cd python
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers tokenizers huggingface_hub
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last command should print `True` and your GPU name.

**3. Download the model (about 3 GB)**

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ../qwen2.5-1.5b
deactivate
cd ..
```

**4. Build the Rust side**

The first build takes 5–10 minutes.

```bash
cd rust
export CUDA_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
cargo build --release
cd ..
```

**5. Run the benchmark**

```bash
./bench.sh
```

It runs Python, lets the GPU cool down, runs Rust, then prints a results table. The full run takes about 5 minutes. To run only one side, use `./bench.sh python` or `./bench.sh rust`.

**6. See the results**

```bash
cat results/summary.md
```

### Windows (PowerShell)

Run these in **Developer PowerShell for VS**, so the Rust build can find the C++ compiler.

**1. Clone the repo**

```powershell
git clone https://github.com/<your-username>/rust-vs-python-gpu.git
cd rust-vs-python-gpu
```

**2. Set up Python**

```powershell
cd python
uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers tokenizers huggingface_hub
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last command should print `True` and your GPU name.

**3. Download the model (about 3 GB)**

```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ..\qwen2.5-1.5b
cd ..
```

Keep the Python environment active for the remaining steps.

**4. Build the Rust side**

The first build takes 5–10 minutes.

```powershell
cd rust
$env:CUDA_COMPUTE_CAP = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Replace(".", "")
cargo build --release
cd ..
```

**5. Run the benchmark**

```powershell
mkdir results
python python\bench.py qwen2.5-1.5b results\python.json
.\rust\target\release\qwen-bench.exe qwen2.5-1.5b results\rust.json
python summarize.py results
```

**6. See the results**

```powershell
type results\summary.md
```

On Windows, the summary includes speed, time to first token and startup time. The GPU telemetry, memory and install size rows need `bench.sh`, which runs on Linux. To get every metric on a Windows machine, follow the Linux steps inside WSL2.

## Project structure

```
rust-vs-python-gpu/
├── bench.sh            # runs both sides and logs GPU metrics (Linux)
├── summarize.py        # turns raw results into a Markdown table
├── python/
│   └── bench.py        # PyTorch benchmark
└── rust/
    ├── Cargo.toml
    └── src/main.rs     # Candle benchmark
```

## Cleanup

**Linux**

```bash
rm -rf python/.venv rust/target qwen2.5-1.5b
uv cache clean
```

**Windows**

You can delete the following folders inside the repo:

- `python\.venv`
- `rust\target`
- `qwen2.5-1.5b`

Then run `uv cache clean` to clear uv's download cache.

## License

MIT
