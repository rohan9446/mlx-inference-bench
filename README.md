# mlx-inference-bench

LLM inference benchmarking on Apple Silicon (MLX), driven by an
OpenAI-compatible load client so the same measurement can later be pointed at
other serving stacks without changing the client.

**Current status:** a standalone M1 characterization. No NVIDIA data collected,
and this is not one arm of a controlled comparison.

## Results

[`results.md`](results.md) — environment, workload, two sweeps, two ablations,
the memory ceiling, repeatability, and limitations.

Qwen3-0.6B bf16 on a MacBook Pro M1 (8GB), thinking disabled, seed 42,
`--prompt-cache-size 1`, concurrency 1:

| ISL | TTFT p50 | Prefill rate | ITL p50 |
|---:|---:|---:|---:|
| 128 | 282 ms | 454 tok/s | 22.19 ms |
| 512 | 553 ms | 925 tok/s | 27.02 ms |
| 2,048 | 2,071 ms | 989 tok/s | 27.19 ms |
| 8,192 | 12,128 ms | 675 tok/s | 47.91 ms |

Four things this measured:

1. **Prefill rate is non-monotonic**, peaking near 2,048 tokens and falling
   beyond it.
2. **Inter-token latency roughly doubles** from 128 to 8,192 tokens, with the
   rise concentrated at the top end rather than spread smoothly.
3. **Prefill batching trades TTFT for ITL.** At concurrency 8, setting
   `--prompt-concurrency 1` halves median TTFT and makes ITL 39% worse. The
   default is tuned for aggregate throughput, not for median time-to-first-token.
4. **The 8GB ceiling is reached through prompt-cache retention**, not model
   size — and how it fails depends on whether swap is available.

## Read the limitations

Section 8 is not boilerplate. In particular: the same workload measured 18%
apart at two points in the same session (§7.2), swap was active for most of the
run (§8.1), and no GPU counters exist on this platform (§8.3), so every causal
statement is interpretation rather than result.

## Why a single load client

`mlx_lm.server` exposes an OpenAI-compatible API, so
[NVIDIA AIPerf](https://github.com/ai-dynamo/aiperf) drives it with the same
load generation, arrival pattern and metric definitions it would use against
vLLM. Only the server under test would change.

AIPerf's GPU telemetry backends are DCGM and pynvml, so on Apple Silicon it
reports `Platform: unknown` and collects no counters.

## Reproducing

The whole session is one script:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
chmod +x run_baseline.sh
caffeinate -i ./run_baseline.sh 2>&1 | tee run_baseline.out
```

Roughly 75 minutes. It runs a preflight probe and aborts before the sweeps if
thinking suppression did not reach the request payload, starts and stops its own
servers, records swap around every run, and writes a manifest with versions and
the resolved model SHA.

**It is not fully unattended.** Fourteen of the fifteen runs complete on their
own. The last one deliberately triggers a Metal OOM, and AIPerf does not recover
after the server's generation thread dies — it hangs and needs Ctrl-C. There is
no watchdog. Everything up to that point is written to disk, and `results.md` §6
describes what the run establishes.

The model revision is resolved and recorded, not pinned — see `results.md` §2.

## Layout

```
results.md           measurements, ablations, limitations
run_baseline.sh      the full session, one command
plot_results.py      figures (data hardcoded, also a record of it)
requirements.lock    pinned Python environment (AIPerf 0.12.0)
results_parity/      raw artifacts, one directory per run, plus server logs
figures/             rendered plots
artifacts/           earlier round, superseded — see results.md
```

## Open questions

Listed in [`results.md`](results.md) §10, with the measurements that motivate
each one. The nearest of them needs no hardware: the per-request TTFT
distributions in `results_parity/*/profile_export.jsonl` would test the batching
interpretation in §5.1 directly.
