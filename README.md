# mlx-inference-bench

LLM inference benchmarking across Apple Silicon (MLX) and NVIDIA (vLLM), driven
by a single OpenAI-compatible load client so the measurement stays constant when
the hardware changes.

**Current status:** Apple Silicon baseline complete. NVIDIA data not yet
collected.

## Results

[`results.md`](results.md) — the full writeup: environment, frozen workload,
both sweeps, the memory ceiling, reproducibility, and a limitations section.

Headline numbers, Qwen3-0.6B bf16 on a MacBook Pro M1 (8GB), concurrency 1:

| ISL | TTFT p50 | Effective prefill rate | ITL p50 |
|---:|---:|---:|---:|
| 128 | 269 ms | 475 tok/s | 22.50 ms |
| 512 | 520 ms | 984 tok/s | 23.33 ms |
| 2,048 | 2,023 ms | 1,012 tok/s | 26.52 ms |
| 8,192 | 12,025 ms | 681 tok/s | 41.60 ms |

Three things this measured:

1. Effective prefill rate is non-monotonic — it peaks around 2,048 tokens and
   falls beyond it.
2. Inter-token latency rises 85% with prompt length. Decode is not independent
   of context.
3. The 8GB ceiling is reached through prompt-cache retention, not model size.

## Why a single load client

Both `mlx_lm.server` and vLLM expose OpenAI-compatible APIs, so
[NVIDIA AIPerf](https://github.com/ai-dynamo/aiperf) can drive every platform in
the comparison. The load generator, arrival pattern, and metric definitions stay
identical across hardware; only the server under test changes.

AIPerf's GPU telemetry backends are DCGM/pynvml, so on Apple Silicon it reports
`Platform: unknown` and collects no counters. That gap is documented in
`results.md` §7.

## Reproducing

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

# terminal 1
mlx_lm.server --model Qwen/Qwen3-0.6B --port 8080 --prompt-cache-size 1

# terminal 2
aiperf profile --model Qwen/Qwen3-0.6B --endpoint-type chat --streaming \
  --url http://localhost:8080 --concurrency 1 \
  --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 --request-count 100
```

`--prompt-cache-size 1` is part of the workload spec, not a tuning choice — see
`results.md` §5.

Note: the runs in `results.md` were collected without a pinned random seed, so
synthetic prompts are not reproducible token-for-token. Seed pinning applies from
the NVIDIA runs onward. See `results.md` §7.

## Layout

```
results.md                  measurements, analysis, limitations
plot_results.py             figures (data hardcoded — also a record of it)
requirements.lock           pinned Python environment
artifacts/                  raw AIPerf exports (csv, json, logs) per run
figures/                    rendered plots
```

## Next

1. vLLM thinking-suppression parity — `--reasoning-parser qwen3` parses
   reasoning output, it does not suppress it. Must be settled before NVIDIA data
   is collected, or ITL and OSL are not measuring the same thing across
   platforms.
2. NVIDIA sweeps (A-series, L-series) with identical AIPerf flags.
3. M5 Pro (~307 GB/s) to test whether the ITL-vs-context curve flattens with
   bandwidth.
4. `vllm-mlx` on the same M1 to separate stack contribution from hardware.
