# LLM Inference on Apple Silicon: M1 Baseline

Measured characterization of `mlx_lm.server` serving Qwen3-0.6B on a MacBook Pro
M1 (8GB unified memory). This is a preliminary Apple Silicon baseline for a
planned cross-platform comparison against vLLM on NVIDIA. No NVIDIA data is
included here yet, and see §2 for why these runs are not yet a valid comparison
arm.

All measurements: 2026-09-19.

---

## 1. Environment

| Component | Version |
|---|---|
| Hardware | MacBook Pro 13", M1, 2020, 8GB unified memory |
| OS | macOS 27.0 |
| Python | 3.12 (Homebrew) |
| mlx | 0.32.2 |
| mlx-lm | 0.31.3 |
| mlx-metal | 0.32.2 |
| transformers | 5.17.0 |
| numpy | 2.5.3 |
| Load client | NVIDIA AIPerf 0.12.0 |

`mlx_lm.server` stamps every response chunk with a `system_fingerprint`, which
gives per-request provenance for free:

```
0.31.3-0.32.2-macOS-27.0-arm64-arm-64bit-applegpu_g13g
```

There is no vLLM equivalent; the vLLM version must be recorded manually when the
NVIDIA runs happen.

---

## 2. Workload specification and known deviations

The specification below is what every machine in the comparison must match.
These runs deviate from it in three ways, documented at the end of this section.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` (HF repo, not an mlx-community conversion) |
| Precision | bf16 |
| Thinking | **NOT disabled in these runs — see below** |
| Output length target | 128 tokens |
| Temperature | 0 |
| Random seed | **not pinned** (see §7) |
| Model revision | **not pinned to a commit SHA** (see §7) |
| Concurrency | 1 (ISL sweep), 1–8 (concurrency sweep) |
| Requests per run | 100 (except ISL 8192, n=20) |

**Server:**

```bash
mlx_lm.server --model Qwen/Qwen3-0.6B --port 8080 --prompt-cache-size 1
```

**Client:**

```bash
aiperf profile --model Qwen/Qwen3-0.6B --endpoint-type chat --streaming \
  --url http://localhost:8080 --concurrency <C> \
  --synthetic-input-tokens-mean <ISL> --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 --request-count 100
```

### Thinking suppression: specified, verified, and then not applied

This is a known defect in these runs, disclosed rather than corrected after the
fact.

The mechanism works. `mlx_lm` has no `--no-think` flag; thinking is suppressed
through the chat template. On `mlx_lm.server` (v0.31.3) the option is
`--chat-template-args '{"enable_thinking": false}'`; over HTTP the per-request
field is `{"chat_template_kwargs": {"enable_thinking": false}}`.

**Neither was applied to the benchmark runs.** The server was started without
`--chat-template-args`, and no AIPerf invocation passed
`chat_template_kwargs`. The committed request payloads
(`artifacts/*/inputs.json`) confirm it:

```json
{"messages": [...], "model": "Qwen/Qwen3-0.6B", "stream": true,
 "max_completion_tokens": 128, "min_tokens": 128, "ignore_eos": true}
```

Qwen3's chat template defaults `enable_thinking` to true, so the generated
tokens in every run here were most likely reasoning tokens rather than answer
text.

**What this affects.** Nothing in the latency mechanics. Decode cost per token
is set by weight and KV-cache traffic, which does not depend on what the token
says; prefill never touches the output at all. The ISL curve, the effective
prefill rate, the memory ceiling, and the concurrency behaviour all stand.

**What it does affect** is the cross-platform comparison, which has not been run
yet. If vLLM suppresses thinking and MLX does not, ITL and OSL are not measuring
the same thing. Every future run on either platform must carry an explicit
thinking setting, and the rendered request payload must be committed as evidence
that it took effect.

**Open parity item:** on the vLLM side, `--reasoning-parser qwen3` *parses*
reasoning output; it does not suppress it. Suppression is
`--default-chat-template-kwargs '{"enable_thinking": false}'` server-side, or
`chat_template_kwargs` per request. Verify with a single request and commit the
payload before collecting anything.

### These runs are not yet a valid comparison arm

Three deviations from the specification above, taken together:

| Specified | As run |
|---|---|
| Thinking disabled | Thinking on (this section) |
| Fixed seed | No `--random-seed` passed (§7.7) |
| Pinned model revision | Loaded from `main` (§7.7) |

The NVIDIA runs will be collected *with* all three applied. That makes them a
different workload, not the same one, so this baseline cannot be set beside them
and called a like-for-like comparison. Two ways forward, and this project takes
the first:

1. **Re-run the M1 baseline once** under the final frozen configuration —
   thinking disabled, `--random-seed` set, model pinned to a commit SHA,
   `--artifact-dir` per configuration. Roughly 45 minutes of machine time, and
   it makes the comparison valid.
2. Present the NVIDIA work as an independent follow-up benchmark rather than a
   comparison.

Until (1) happens, everything here is a preliminary characterization of MLX on
M1, not one half of a controlled experiment. The findings in §3–§5 are about
latency mechanics and are unaffected by the deviations; what the deviations
block is the *comparison*, not the *measurements*.

---

## 3. ISL sweep (concurrency 1)

Note on the prefill column: this is the *effective per-request prefill rate*,
derived as ISL / TTFT. TTFT includes scheduling and queueing, so this is a
user-visible rate, not a direct engine-level count of prefill tokens executed
per second. The distinction matters under concurrency (§4).

| ISL | TTFT p50 (ms) | TTFT avg (ms) | Effective prefill rate tok/s (p50) | ITL p50 (ms) | ITL avg (ms) |
|---:|---:|---:|---:|---:|---:|
| 128 | 269.28 | 281.76 | 475.25 | 22.50 | 22.55 |
| 512 | 520.30 | 521.57 | 984.04 | 23.33 | 23.42 |
| 2048 | 2,023.14 | 2,025.98 | 1,012.28 | 26.52 | 26.59 |
| 8192 | 12,024.67 | 12,094.53 | 681.19 | 41.60 | 42.28 |

ISL 8192 is n=20; all others n=100.

**Evidence status:** these four rows are transcribed from AIPerf's console
output at the time of each run. Their raw exports were **not** preserved — AIPerf
names its artifact directory by concurrency, and all four ran at concurrency 1,
so each overwrote the last. The surviving `concurrency1` directory holds the
final run written to it (an ISL 512 repeat with `min_tokens`/`ignore_eos` set,
TTFT p50 520.97, ITL p50 23.32). The table's ISL 512 row comes from the earlier,
unpreserved run at 520.30 / 23.33; the two agree to 0.13%, which is the
reproducibility check in §6 but is not a substitute for the missing exports.
Treat §3 as reported-but-not-independently-verifiable, and §4 as verifiable.
Future runs use `--artifact-dir` per configuration.

### Finding 1 — effective prefill rate is non-monotonic

475 → 984 → 1,012 → 681 tok/s.

The rate climbs steeply from 128 to 512, plateaus around 2048, then **falls** at
8192. The measurement is the shape; the mechanism below is interpretation, since
no Apple GPU utilization or bandwidth counters were collected (§7.3).

A reading consistent with the data: at short prompts, fixed per-request and
kernel-launch overhead is amortized over too few tokens, and the ramp is that
overhead being diluted. At 8192 the fall is consistent with attention cost
growing faster than linearly in sequence length and overtaking whatever
utilization gain remains. Neither half of that is established here — memory
pressure at 8192 is an equally live candidate for the fall, given §5.

What is measured: an efficiency peak around 2048 on this hardware, with a real
penalty beyond it.

Practical consequence: a per-token cost model calibrated at ISL 128 overstates
prefill cost by roughly 2× at ISL 512.

### Finding 2 — TTFT at 8192 is 49% worse than linear

At the observed peak of 1,012 tok/s, 8192 tokens should prefill in ~8.09 s.
Measured: 12.02 s. The gap is the Finding 1 falloff expressed as latency rather
than as a rate; the same interpretive caveat applies to its cause.

### Finding 3 — inter-token latency rises 85% with prompt length

22.50 → 41.60 ms p50, from ISL 128 to ISL 8192.

Decode is conventionally described as independent of prompt length. It is not.
Every decode step re-reads the entire KV cache, so the per-token memory traffic
grows with context.

Rough check for Qwen3-0.6B (28 layers, 8 KV heads, head_dim 128, bf16):
~0.11 MB of KV per token.

| ISL | KV re-read per decode step | Predicted ITL delta @ 68 GB/s | Measured delta |
|---:|---:|---:|---:|
| 128 | ~14 MB | baseline | baseline |
| 8192 | ~900 MB | ~+13 ms | +19.1 ms |

Same order of magnitude, correct direction. The residual is attention compute
and scheduling overhead on top of the raw streaming cost.

This falsifies the naive roofline prediction this project started from, which
treated decode as a fixed per-token cost set by weight bandwidth alone.

**Testable prediction:** the M5 Pro (~307 GB/s, ~4.5× this machine) should show
a substantially flatter ITL-vs-context curve. An L4 (~300 GB/s) is matched on
bandwidth with completely different architecture, which makes it a controlled
comparison rather than only a faster number.

---

## 4. Concurrency sweep (ISL 512, OSL 128)

| Concurrency | TTFT p50 (ms) | ITL p50 (ms) | Output tok/s (all users) | Per-user decode tok/s | Error rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 520.30 | 23.33 | 36.56 | 43.08 | 0% |
| 2 | 979.03 | 44.12 | 38.70 | 22.76 | 0% |
| 4 | 1,787.34 | 48.24 | 53.35 | 20.84 | 0% |
| 8 | 3,400.50 | 53.70 | 96.02 | 18.62 | 2% |

### Finding 4 — decode scales with concurrency; prefill capacity saturates

The two phases respond very differently to added load.

**Decode scales.** From c=2 to c=8, concurrency rose 4× while ITL rose only 22%
(44.12 → 53.70 ms) and total output throughput rose 2.5×. Per-user decode held
roughly flat (22.76 → 18.62 tok/s). Active decode throughput climbed
45 → 83 → 150 tok/s.

The c=1 → c=2 step is a one-time cost, not the trend: ITL doubles and throughput
is flat. Everything after that amortizes.

**Prefill does not.** TTFT p50 scales ~1.9× per doubling: 520 → 979 → 1,787 →
3,400 ms. Effective per-request prefill rate falls almost exactly
proportionally (984 → 523 → 287 → 151 tok/s), while the summed rate across
users stays pinned around 1,000–1,200 tok/s — which is approximately the
*single-request* peak measured in §3 (1,012 tok/s at ISL 2048).

That is the key observation: **prefill was already at its ceiling at
concurrency 1.** There is no spare prefill capacity for added concurrency to
exploit, so additional requests convert directly into TTFT.

Net over the range: 8× concurrency buys 2.6× total output throughput. TTFT
degraded far more sharply in relative terms than ITL, 6.5× against 2.3×. In
absolute terms both costs are real — at ~127 output tokens, the ITL increase
adds roughly 3.9 s to each request's decode phase, against a 2.9 s increase in
TTFT — so this is a difference in scaling behaviour, not a case of decode being
free.

**What this does not establish.** An earlier draft of this document claimed
prompts are processed one at a time. That claim is withdrawn — it was inferred
from latency scaling, not measured. `mlx_lm.server` exposes
`--prompt-concurrency` ("prompts in parallel") and a decode concurrency setting,
neither of which was varied here; only `--prompt-cache-size` was set. The data
show that prefill *capacity* saturates at low concurrency. They do not identify
the cause, which could be GPU compute saturation, memory bandwidth, scheduler
policy, chunked-prefill behavior, batching efficiency, or serialization
somewhere in the stack.

Distinguishing those requires a controlled sweep — hold concurrency fixed and
vary `--prompt-concurrency` — which has not been run.

---

## 5. Memory ceiling

### The OOM

ISL 8192 with default server settings fails with a hard Metal allocation error,
not swap thrashing:

```
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

Server logs identify the cause. `mlx_lm.server` retains KV caches for the last
N requests (default N=10):

```
Prompt Cache: 10 sequences, 2.50 GB     # at ISL 2048 — holds
Prompt Cache: 10 sequences, 3.21 GB     # at ISL 8192 — dies on request 2
```

A single 8192-token request completes fine. Ten retained caches do not. The
binding constraint is prompt-cache retention, not model size.

### The fix

```bash
--prompt-cache-size 1     # max distinct KV caches held
--prompt-cache-bytes N    # alternative: byte budget
```

With `--prompt-cache-size 1`, ISL 8192 runs 20/20 requests clean.

### Retention was pure overhead here

Re-measuring the shorter lengths under both settings:

| ISL | Metric | cache=10 | cache=1 |
|---:|---|---:|---:|
| 128 | TTFT p50 | 273.53 | 269.28 |
| 128 | Prefill tok/s | 467.87 | 475.25 |
| 128 | ITL p50 | 22.86 | 22.50 |
| 512 | TTFT p50 | 527.89 | 520.30 |
| 512 | TTFT std | 51.14 | 6.95 |

Within noise on the means, and **variance drops sharply** at 512. AIPerf
generates distinct synthetic prompts, so there are no prefix hits to reuse — the
retained caches consumed memory and bought nothing. For benchmark workloads
specifically, default retention is overhead that eventually kills the run.

This does not generalize to real serving, where prefix reuse is common and the
cache earns its memory.

### Concurrency ceiling

c=8 returned a 2% error rate (98/100 succeeded). The suspected cause is memory
pressure — eight live KV caches plus weights against 8GB — but this is **not
confirmed**: the failed-request logs were not inspected, so it is not
established that these failures share the Metal OOM signature above. Treated
here as suspected memory pressure at the practical edge of the tested
configuration.

---

## 6. Reproducibility

The ISL 512 / c=1 configuration was measured twice, ~25 minutes apart, in
separate AIPerf invocations against the same server process:

| Run | TTFT p50 (ms) | ITL p50 (ms) |
|---|---:|---:|
| First | 520.30 | 23.33 |
| Second | 520.97 | 23.32 |

0.13% and 0.04% apart.

### Warm-up matters enormously

Prompt throughput across three successive cold-to-warm runs on identical
hardware, same model, same request:

```
8.337 → 77.476 → 105.921 tok/s
```

A 9–13× swing from kernel compilation and cache warming alone. Every number in
this document comes from a warmed server. Benchmarks that report a first run are
reporting compilation time.

---

## 7. Limitations

Stated rather than hidden.

1. **Co-located load client.** AIPerf runs on the same machine as the server —
   only one machine was available. The load client competes for CPU with the
   server process. Affects all numbers here equally, so trends are sound;
   absolute values are pessimistic by an unmeasured margin.

2. **Output length is not pinned.** OSL comes back 126.97 ± 0.22 rather than a
   clean 128. `mlx_lm.server` ignores AIPerf's `min_tokens` and `ignore_eos`
   extra-inputs — verified by passing them explicitly and observing no change
   (126.99, min 126, max 127). `max_tokens` caps but does not pin. These flags
   were dropped rather than left in, because if vLLM *does* honor them, keeping
   them would create a silent asymmetry between platforms — worse than no
   pinning at all.

3. **No GPU telemetry.** AIPerf prints `Platform: unknown` and
   `No GPU telemetry data collected during the benchmarking run.` on every run.
   Its telemetry backends are DCGM/pynvml, which are NVIDIA-only. No
   utilization, power, or memory-bandwidth counters were captured on Apple
   Silicon. Joules-per-token — the intended cross-platform axis — is therefore
   not yet measurable here. An Apple Silicon telemetry backend would need
   `powermetrics` and `mx.get_peak_memory`.

4. **`--prompt-cache-size` has no vLLM equivalent.** The comparison will have a
   configuration asymmetry that must be disclosed, not papered over.

5. **ISL 8192 is n=20**, not n=100, because each request takes ~21 s.

6. **One model, one precision.** Qwen3-0.6B at bf16. Nothing here extrapolates
   to 7B+ models, where weight streaming dominates differently.

7. **The workload is not byte-identical, despite §2 calling for it.** No random
   seed was passed to AIPerf, so synthetic prompts are not reproducible across
   invocations. The model and tokenizer were loaded from the HF repo without
   pinning commit SHAs, and the AIPerf version was not recorded. These runs
   cannot be reproduced token-for-token after the fact. What supports them
   instead is sample size (n=100 per point) and the ISL 512 repeat agreeing to
   0.13% — evidence of stability, not of reproducibility. The package set is
   pinned after the fact in `requirements.lock`; the seed and the model revision
   are not recoverable. Seed and revision pinning apply from the NVIDIA runs
   onward.

8. **Repeatability was tested within one server process, not across restarts.**
   The §6 repeat used the same running server. Fresh-process repeats, which
   would catch start-up and allocation variance, were not run.

9. **The prefill saturation cause is unidentified.** See §4 — capacity
   saturation is measured; its mechanism is not.

10. **Thinking was not suppressed.** See §2. The generated tokens were most
    likely reasoning tokens. This does not affect the latency mechanics measured
    here, but it means the workload as run differs from the workload as
    specified, and it must be fixed before any cross-platform comparison.

11. **The ISL-sweep raw exports were overwritten.** See §3. Only the
    concurrency-sweep artifacts are independently verifiable. The ISL rows are
    reported from console output.

12. **Causal language is interpretation.** Findings 1 and 2 describe measured
    shapes; the explanations offered for them (overhead amortization, attention
    cost, memory pressure) are hypotheses consistent with the data, not results.
    No GPU counters were collected on this platform.

---

## 8. Evidence

What is committed, and what is not.

**Committed** — one directory per concurrency level, from the concurrency sweep
in §4:

```
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency{1,2,4,8}/
    inputs.json                  rendered request payloads
    profile_export.jsonl         per-request records
    profile_export_aiperf.{csv,json}
    profile_export_console.txt
    logs/aiperf.log
```

`concurrency{2,4,8}` back the §4 rows directly. `concurrency1` holds the last
run written to it, not the run quoted in §3 or the c=1 row of §4 — see the
evidence note in §3.

The Python environment is pinned in `requirements.lock` (AIPerf 0.12.0).

**Not committed, and not recoverable:**

- Raw exports for ISL 128 / 2048 / 8192, and for the ISL 512 run quoted in the
  tables (overwritten).
- The cache=10 vs cache=1 comparison runs in §5.
- The warm-up sequence in §6.
- `mlx_lm.server` stdout with the prompt-cache growth lines and the Metal OOM
  traceback. The error string in §5 is transcribed verbatim from the terminal.
- Model and tokenizer commit SHAs.
- Failure-specific logs for the two c=8 errors (the c=8 export records them as
  connection resets).

Claims resting on uncommitted evidence are marked as such where they appear.

---

## 9. Next

0. **Thinking suppression on both platforms, verified by committed payload.**
   One request each, `inputs.json` inspected, before any comparison data is
   collected. This supersedes the previous "vLLM-only" framing of this item —
   MLX needs it too (§2).
1. **Re-run the full M1 baseline under the frozen configuration** — thinking
   disabled, `--random-seed` set, model pinned to a commit SHA, `--artifact-dir`
   per configuration. This is what turns §3–§5 from a preliminary
   characterization into a comparison arm, and it also restores the ISL raw
   exports lost to overwriting.
2. NVIDIA sweeps — A-series and L-series — with identical AIPerf flags. Record
   exact SKUs: bandwidth is the axis, so "A100" without 40GB/80GB is not a
   data point.
3. M5 Pro (~307 GB/s) — confirm macOS ≥ 26.2 first, or MLX silently falls back
   and skips the Neural Accelerators, which would quietly invalidate the prefill
   comparison.
4. `vllm-mlx` on this same M1 — isolates stack contribution from hardware, since
   the silicon is held constant.
5. Pin seed, AIPerf version, and model/tokenizer commit SHAs from the first
   NVIDIA run onward.

Not planned: further M1 collection. The `--prompt-concurrency` sweep that would
identify the prefill saturation mechanism (§4) is left as an open question
rather than an answered one.
