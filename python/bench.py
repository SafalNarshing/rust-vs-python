"""PyTorch side of the Rust vs Python GPU benchmark.

Usage: python bench.py <model_dir> <output_json>

Runs Qwen2.5-1.5B-Instruct in BF16 on CUDA with greedy decoding, generating
exactly MAX_NEW_TOKENS tokens per prompt (EOS ignored so both sides do the same
amount of work). Prompts, chat template and tokenizer are identical to the Rust
version.
"""
import time

T_START = time.perf_counter()  # measured before importing torch on purpose

import json
import statistics
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

MAX_NEW_TOKENS = 256
WARMUP_TOKENS = 32
SYSTEM = "You are a helpful assistant."
PROMPTS = [
    "Explain what a GPU kernel is in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between Rust and Python?",
    "Describe how a transformer model processes a sentence.",
    "Give me five tips for writing clean code.",
    "Explain the concept of memory safety.",
    "What is the role of an operating system?",
    "Summarize the history of the internet in a few paragraphs.",
    "How does a hash map work internally?",
    "Write a poem about the mountains of Nepal.",
    "Explain gradient descent to a beginner.",
    "What makes a good API design?",
    "Describe the water cycle step by step.",
    "What are the pros and cons of microservices?",
    "Explain what a compiler does.",
    "How do vaccines train the immune system?",
    "Write a product description for a smart water bottle.",
    "What is the difference between concurrency and parallelism?",
    "Explain how public key cryptography works.",
    "Describe a typical day in the life of a software engineer.",
]


def chat(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, n_tokens: int):
    ids = tokenizer.encode(chat(prompt), add_special_tokens=False).ids
    input_ids = torch.tensor([ids], device="cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[0, -1].float().argmax().item()  # .item() syncs the GPU
    ttft = time.perf_counter() - t0

    generated = [next_id]
    t1 = time.perf_counter()
    for _ in range(n_tokens - 1):
        step = torch.tensor([[next_id]], device="cuda")
        out = model(input_ids=step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[0, -1].float().argmax().item()
        generated.append(next_id)
    decode_s = time.perf_counter() - t1

    return {
        "prompt_tokens": len(ids),
        "ttft_ms": ttft * 1000,
        "decode_tps": (n_tokens - 1) / decode_s,
        "total_s": ttft + decode_s,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }


def main():
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../qwen2.5-1.5b")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "results_python.json")

    if not torch.cuda.is_available():
        sys.exit("CUDA not available to PyTorch. Check nvidia-smi and your torch install.")

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - T_START
    print(f"[python] model ready in {load_s:.2f}s on {torch.cuda.get_device_name(0)}")

    generate(model, tokenizer, PROMPTS[0], WARMUP_TOKENS)  # warm-up, not timed
    torch.cuda.reset_peak_memory_stats()

    runs = []
    for i, p in enumerate(PROMPTS, 1):
        r = generate(model, tokenizer, p, MAX_NEW_TOKENS)
        runs.append(r)
        print(f"[python] {i:2d}/{len(PROMPTS)}  ttft {r['ttft_ms']:6.1f} ms  decode {r['decode_tps']:6.1f} tok/s")

    result = {
        "impl": "python",
        "framework": f"PyTorch {torch.__version__}",
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bf16",
        "load_s": load_s,
        "ttft_ms_median": statistics.median(r["ttft_ms"] for r in runs),
        "decode_tps_median": statistics.median(r["decode_tps"] for r in runs),
        "generation_s_total": sum(r["total_s"] for r in runs),
        "tokens_generated": MAX_NEW_TOKENS * len(PROMPTS),
        "torch_peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
        "sample_output": runs[0]["text"][:300],
        "runs": [{k: v for k, v in r.items() if k != "text"} for r in runs],
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[python] median decode {result['decode_tps_median']:.1f} tok/s, "
          f"median TTFT {result['ttft_ms_median']:.1f} ms -> {out_path}")


if __name__ == "__main__":
    main()
