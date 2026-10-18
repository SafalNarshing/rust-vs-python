"""Turns the raw benchmark output in results/ into a Markdown table for the blog.
Standard library only. Usage: python3 summarize.py results/
"""
import csv
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def gpu_stats(name):
    path = out / f"gpu_{name}.csv"
    if not path.exists():
        return {}
    rows = []
    for r in csv.reader(path.open(encoding="utf-8")):
        try:
            rows.append([float(x) for x in r[1:]])  # util, mem, power, clock, temp
        except (ValueError, IndexError):
            continue
    if not rows:
        return {}
    baseline = float((out / f"baseline_{name}.txt").read_text(encoding="utf-8").strip() or 0)
    busy = [r for r in rows if r[0] > 5] or rows  # samples where the GPU was actually working
    return {
        "vram_peak_mb": max(r[1] for r in rows) - baseline,
        "util_avg": sum(r[0] for r in busy) / len(busy),
        "power_avg_w": sum(r[2] for r in busy) / len(busy),
        "clock_avg_mhz": sum(r[3] for r in busy) / len(busy),
        "temp_max_c": max(r[4] for r in rows),
    }


def time_stats(name):
    path = out / f"time_{name}.txt"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    wall = re.search(r"Elapsed \(wall clock\) time.*: ([\d:.]+)", text)
    secs = 0.0
    if wall:
        for part in wall.group(1).split(":"):
            secs = secs * 60 + float(part)
    return {"ram_peak_mb": int(rss.group(1)) / 1024 if rss else None, "wall_s": secs}


sizes = {}
if (out / "sizes.txt").exists():
    for line in (out / "sizes.txt").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            k, v = line.split()
            sizes[k] = int(v or 0) / 2**20

data = {}
for name in ("python", "rust"):
    p = out / f"{name}.json"
    if p.exists():
        data[name] = {**json.loads(p.read_text(encoding="utf-8")), **gpu_stats(name), **time_stats(name)}

if not data:
    sys.exit("No results found.")


def cell(name, key, fmt):
    v = data.get(name, {}).get(key)
    return fmt.format(v) if isinstance(v, (int, float)) else "-"


rows = [
    ("Decode speed (tokens/s, median)", "decode_tps_median", "{:.1f}"),
    ("Time to first token (ms, median)", "ttft_ms_median", "{:.1f}"),
    ("Generation time, 20 prompts (s)", "generation_s_total", "{:.1f}"),
    ("Startup to model ready (s)", "load_s", "{:.2f}"),
    ("Total wall time (s)", "wall_s", "{:.1f}"),
    ("Peak VRAM (MB)", "vram_peak_mb", "{:.0f}"),
    ("Peak system RAM (MB)", "ram_peak_mb", "{:.0f}"),
    ("Avg GPU utilization (%)", "util_avg", "{:.0f}"),
    ("Avg GPU power (W)", "power_avg_w", "{:.0f}"),
    ("Max GPU temperature (C)", "temp_max_c", "{:.0f}"),
]

gpu = data.get("python", {}).get("gpu", "NVIDIA GPU")
lines = [f"### Qwen2.5-1.5B-Instruct, BF16, greedy, 20 prompts x 256 tokens on {gpu}", "",
         "| Metric | PyTorch (Python) | Candle (Rust) |", "| --- | --- | --- |"]
for label, key, fmt in rows:
    lines.append(f"| {label} | {cell('python', key, fmt)} | {cell('rust', key, fmt)} |")
if sizes:
    lines.append(f"| Install size (MB) | {sizes.get('python_venv_bytes', 0):.0f} (venv incl. bundled CUDA libs) "
                 f"| {sizes.get('rust_binary_bytes', 0):.1f} (binary) + {sizes.get('cuda_toolkit_bytes', 0):.0f} (system CUDA) |")
lines += ["", "Framework versions: " + ", ".join(d.get("framework", "?") for d in data.values())]
for name, d in data.items():
    lines += ["", f"Sample output ({name}): {d.get('sample_output', '')[:200]!r}"]

text = "\n".join(lines)
print(text)
(out / "summary.md").write_text(text + "\n", encoding="utf-8")
print(f"\nSaved to {out / 'summary.md'}")
