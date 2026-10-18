//! Candle (Rust) side of the Rust vs Python GPU benchmark.
//!
//! Usage: qwen-bench <model_dir> <output_json>
//!
//! Same model, prompts, chat template, tokenizer, BF16 precision and greedy
//! decoding as python/bench.py. Generates exactly MAX_NEW_TOKENS per prompt.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const MAX_NEW_TOKENS: usize = 256;
const WARMUP_TOKENS: usize = 32;
const SYSTEM: &str = "You are a helpful assistant.";
const PROMPTS: [&str; 20] = [
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
];

struct Run {
    prompt_tokens: usize,
    ttft_ms: f64,
    decode_tps: f64,
    total_s: f64,
    text: String,
}

fn chat(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn argmax_last(logits: &Tensor) -> Result<u32> {
    // logits: (1, 1, vocab). to_scalar copies to host, which also syncs the GPU.
    Ok(logits
        .squeeze(0)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .argmax(0)?
        .to_scalar::<u32>()?)
}

fn generate(
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    n_tokens: usize,
) -> Result<Run> {
    model.clear_kv_cache();
    let ids = tokenizer
        .encode(chat(prompt), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    device.synchronize()?;
    let t0 = Instant::now();
    let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let mut next = argmax_last(&logits)?;
    let ttft = t0.elapsed().as_secs_f64();

    let mut generated = vec![next];
    let mut pos = ids.len();
    let t1 = Instant::now();
    for _ in 1..n_tokens {
        let step = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&step, pos)?;
        pos += 1;
        next = argmax_last(&logits)?;
        generated.push(next);
    }
    let decode_s = t1.elapsed().as_secs_f64();

    Ok(Run {
        prompt_tokens: ids.len(),
        ttft_ms: ttft * 1000.0,
        decode_tps: (n_tokens - 1) as f64 / decode_s,
        total_s: ttft + decode_s,
        text: tokenizer.decode(&generated, true).map_err(anyhow::Error::msg)?,
    })
}

fn main() -> Result<()> {
    let t_start = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let model_dir = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("../qwen2.5-1.5b"));
    let out_path = PathBuf::from(args.get(2).map(String::as_str).unwrap_or("results_rust.json"));

    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        bail!("CUDA device not available. Build with the default `cuda` feature and check nvidia-smi.");
    }

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let cfg: Config = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let mut weights: Vec<PathBuf> = std::fs::read_dir(&model_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    weights.sort();
    if weights.is_empty() {
        bail!("no .safetensors files found in {}", model_dir.display());
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::BF16, &device)? };
    let mut model = ModelForCausalLM::new(&cfg, vb)?;
    device.synchronize()?;
    let load_s = t_start.elapsed().as_secs_f64();
    println!("[rust] model ready in {load_s:.2}s");

    generate(&mut model, &tokenizer, &device, PROMPTS[0], WARMUP_TOKENS)?; // warm-up, not timed

    let mut runs = Vec::with_capacity(PROMPTS.len());
    for (i, p) in PROMPTS.iter().enumerate() {
        let r = generate(&mut model, &tokenizer, &device, p, MAX_NEW_TOKENS)?;
        println!(
            "[rust] {:2}/{}  ttft {:6.1} ms  decode {:6.1} tok/s",
            i + 1,
            PROMPTS.len(),
            r.ttft_ms,
            r.decode_tps
        );
        runs.push(r);
    }

    let result = json!({
        "impl": "rust",
        "framework": "Candle 0.11",
        "dtype": "bf16",
        "load_s": load_s,
        "ttft_ms_median": median(runs.iter().map(|r| r.ttft_ms).collect()),
        "decode_tps_median": median(runs.iter().map(|r| r.decode_tps).collect()),
        "generation_s_total": runs.iter().map(|r| r.total_s).sum::<f64>(),
        "tokens_generated": MAX_NEW_TOKENS * PROMPTS.len(),
        "sample_output": runs[0].text.chars().take(300).collect::<String>(),
        "runs": runs.iter().map(|r| json!({
            "prompt_tokens": r.prompt_tokens,
            "ttft_ms": r.ttft_ms,
            "decode_tps": r.decode_tps,
            "total_s": r.total_s,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&result)?)?;
    println!(
        "[rust] median decode {:.1} tok/s, median TTFT {:.1} ms -> {}",
        result["decode_tps_median"].as_f64().unwrap_or(0.0),
        result["ttft_ms_median"].as_f64().unwrap_or(0.0),
        out_path.display()
    );
    Ok(())
}
