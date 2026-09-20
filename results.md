# LLM Inference on Apple Silicon: M1 Characterization

Measured characterization of `mlx_lm.server` serving Qwen3-0.6B on a MacBook Pro
M1 (8GB unified memory), with NVIDIA AIPerf as the load client.

All numbers here come from a single unattended session on 2026-09-19, 21:08–22:54
local, under one configuration, with thinking disabled, a fixed seed and a
separate artifact directory per run. An earlier round on the same machine is
referenced only where the two disagree; it is not mixed into any table.

This is a standalone report. It is not one arm of a controlled comparison, and
no NVIDIA data is included.

---

## 1. Environment

| Component | Version |
|---|---|
| Hardware | MacBook Pro 13", M1, 2020, 8GB unified memory |
| OS | macOS 27.0 (Darwin 27.0.0 arm64) |
| Python | 3.12.14 (Homebrew) |
| mlx | 0.32.2 |
| mlx-lm | 0.31.3 |
| mlx-metal | 0.32.2 |
| transformers | 5.17.0 |
| numpy | 2.5.3 |
| Load client | NVIDIA AIPerf 0.12.0 |

Full package set in `requirements.lock`. `mlx_lm.server` stamps every response
chunk with `system_fingerprint`, giving per-request provenance:

```
0.31.3-0.32.2-macOS-27.0-arm64-arm-64bit-applegpu_g13g
```

---

## 2. Workload

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B`, revision resolved and recorded, **not pinned** |
| Resolved SHA | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Precision | bf16 |
| Thinking | disabled, verified in the rendered payload |
| Temperature | 0, sent explicitly |
| Random seed | 42 |
| Output length target | 128 tokens |
| Requests per run | 100 (ISL 8192: 20; warm-up: 10; probe: 5) |

**Server:**

```bash
mlx_lm.server --model Qwen/Qwen3-0.6B --port 8080 \
  --prompt-cache-size 1 \
  --chat-template-args '{"enable_thinking": false}'
```

**Client:**

```bash
aiperf profile --model Qwen/Qwen3-0.6B --endpoint-type chat --streaming \
  --url http://localhost:8080 --concurrency <C> \
  --synthetic-input-tokens-mean <ISL> --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --extra-inputs '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  --random-seed 42 --request-count <N> \
  --artifact-dir results_parity/<name>
```

Server defaults worth recording, since two of them are varied in §5:
`--prompt-cache-size 10`, `--prompt-concurrency 8`, `--decode-concurrency 32`.

**On the model revision.** The script queries the HF API for what `main` resolves
to and records the SHA, but `mlx_lm.server` has no revision flag and still loads
`main`. That is provenance, not pinning: if the repo moves, a future run of this
script loads different weights and the recorded SHA will no longer match what
was served. Pinning properly means downloading the snapshot at that SHA, serving
the local path, and passing `--tokenizer-revision <sha>` to AIPerf, which 0.12.0
supports. Not done here.

### Thinking suppression, verified rather than assumed

A previous round on this machine specified thinking as disabled and never
applied it — the flag was tested interactively and omitted from every benchmark
command. This session checks it at the request level before collecting anything:
a 5-request probe, then an assertion against the rendered `inputs.json`.

```
preflight: enable_thinking in payload = True
preflight: temperature in payload     = True
preflight: OSL avg = 127.8  min = 127.0
```

Output length did not collapse with thinking off: 127.8 against a 128 cap. On
synthetic prompts the model runs to the cap either way, so these runs are
broadly comparable to the earlier round despite the different generation mode.

---

## 3. ISL sweep (concurrency 1)

The prefill column is `ISL / TTFT`, the user-visible rate. TTFT includes
scheduling, so it is not an engine-level count of prefill tokens executed per
second. That distinction matters in §4 and §5.

| ISL | TTFT p50 (ms) | Prefill rate (tok/s) | ITL p50 (ms) | OSL avg |
|---:|---:|---:|---:|---:|
| 128 | 282.19 | 454 | 22.19 | 123.15 |
| 512 | 553.34 | 925 | 27.02 | 127.47 |
| 2,048 | 2,070.65 | 989 | 27.19 | 124.95 |
| 8,192 | 12,127.51 | 675 | 47.91 | 127.95 |

ISL 8192 is n=20; the rest n=100.

### Finding 1 — prefill rate is non-monotonic, peaking near 2,048

454 → 925 → 989 → 675 tok/s. This reproduces the earlier round (475 → 984 →
1,012 → 681) within about 4% at every point, which is the strongest agreement
between the two rounds anywhere in this document.

The measurement is the shape. The mechanism is interpretation, since no Apple
GPU utilization or bandwidth counters were collected (§8.3). A reading
consistent with the data: at short prompts fixed per-request overhead is
amortized over too few tokens, and the ramp is that overhead being diluted; the
fall at 8,192 is consistent with attention cost growing faster than linearly.
Memory pressure is an equally live candidate for the fall — swap was in use by
then (§8.1).

### Finding 2 — TTFT at 8,192 is 47% worse than linear

At the observed peak of 989 tok/s, 8,192 tokens should prefill in ~8.28 s.
Measured: 12.13 s. The same falloff as Finding 1, expressed as latency.

### Finding 3 — ITL is flat to 2,048, then rises sharply

22.19 → 27.02 → 27.19 → 47.91 ms. From 128 to 8,192 that is +116%.

**This is not the smooth curve the earlier round suggested,** and the difference
matters. Between 512 and 2,048 the context grows 4× and ITL does not move at all
(27.02 → 27.19). Between 2,048 and 8,192 it grows 4× again and ITL rises 76%.

A pure KV-cache-traffic model predicts monotonic growth throughout, so it does
not fit the flat middle. Whatever drives the 8,192 result is either strongly
non-linear in context length or is something other than cache streaming — memory
pressure being the obvious candidate, since swap was active for the 2,048 and
8,192 runs and not for 128 and 512.

Part of the low-end variation is also session drift rather than context. See
§7.2: ITL at a fixed ISL of 512 measured 22.81–23.26 ms early in the session and
27.02 ms an hour later. That is the same size as the 128→512 step in this table.
The robust claim is the sharp rise at 8,192, not the gradual ramp below it.

For reference, the earlier round reported 22.50 → 23.33 → 26.52 → 41.60 and was
described as a smooth 85% rise. Both rounds agree that ITL roughly doubles from
128 to 8,192. They disagree on the shape in between, and the drift measurement
explains why that middle is unreliable.

---

## 4. Concurrency sweep (ISL 512, server defaults)

| Concurrency | TTFT p50 (ms) | ITL p50 (ms) | Prefill rate (tok/s) | Errors |
|---:|---:|---:|---:|---:|
| 1 | 553.34 | 27.02 | 925 | 0% |
| 2 | 990.06 | 47.19 | 517 | 0% |
| 4 | 1,647.63 | 54.88 | 311 | 0% |
| 8 | 3,316.97 | 61.35 | 154 | 2% |

### Finding 4 — TTFT degrades much faster than ITL

From 1 to 8: TTFT ×6.0, ITL ×2.3.

Per-request prefill rate falls steeply with concurrency (925 → 517 → 311 → 154
tok/s), while the summed rate across users rises only modestly, from about 925
to about 1,232 tok/s — roughly 33% for 8× the offered load, and close to the
single-request peak in §3. Aggregate prefill capacity is therefore nearly
exhausted at low concurrency; most of the additional offered load converts into
TTFT rather than into completed prefill work.

Decode behaves differently. ITL rises 27 → 47 ms on the first doubling and then
only 47 → 61 across the next two, so after the initial step the decode cost of
extra concurrency amortizes.

In absolute terms both costs are real. At ~127 output tokens, the ITL increase
adds roughly 4.4 s to each request's decode phase against a 2.8 s increase in
TTFT. This is a difference in scaling behaviour, not a case of decode being
free.

The earlier round measured ×6.5 and ×2.3 for the same quantities. Close
agreement.

---

## 5. Ablations

### 5.1 Prefill parallelism: `--prompt-concurrency` 1 vs the default 8

This is the experiment the previous round listed as an open question. Request
concurrency is held fixed; only the server's prefill batching changes.

| Config | TTFT p50 (ms) | ITL p50 (ms) | Prefill rate (tok/s) | Errors |
|---|---:|---:|---:|---:|
| conc 4, prompt-concurrency 8 | 1,647.63 | 54.88 | 311 | 0% |
| conc 4, prompt-concurrency 1 | 1,587.19 | 57.92 | 323 | 0% |
| conc 8, prompt-concurrency 8 | 3,316.97 | 61.35 | 154 | 2% |
| conc 8, prompt-concurrency 1 | **1,662.65** | **85.09** | 308 | 2% |

**Finding 5 — prefill batching trades TTFT for ITL, and at concurrency 8 the
trade is severe.** Turning prefill batching off halves median TTFT (3,317 →
1,663 ms) and makes ITL 39% worse (61.35 → 85.09 ms). At concurrency 4 the two
settings are within noise.

An interpretation consistent with the numbers, not established by them: with
`--prompt-concurrency 8` the server gathers eight prompts and completes them
together, so every request waits for the whole batch and all eight TTFTs land
late. With `--prompt-concurrency 1` prompts are processed one at a time and
completions stagger, so the median request sees its first token much sooner
while later requests wait longer. Total prefill work is unchanged, which is why
the median moves so much more than the aggregate.

That reading also fits §4 without contradicting it: aggregate prefill capacity is
close to its ceiling either way, and the batching policy decides how the
resulting wait is distributed across requests. Confirming it requires the
per-request TTFT distributions, which are in the committed
`profile_export.jsonl` files but were not analyzed here.

Practical consequence: on this hardware the default is tuned for aggregate
throughput at the cost of median time-to-first-token, and an interactive
workload at high concurrency would likely prefer `--prompt-concurrency 1`.

### 5.2 Prompt cache size: default 10 vs 1, at ISL 512

| Config | TTFT p50 (ms) | ITL p50 (ms) | Prefill rate (tok/s) |
|---|---:|---:|---:|
| `--prompt-cache-size 1` | 553.34 | 27.02 | 925 |
| default (10) | 540.27 | 24.31 | 948 |

No meaningful difference, and the gap is smaller than the session drift measured
in §7.2. That is the expected result: AIPerf generates distinct synthetic
prompts, so there are no prefix hits and the retained caches cannot help. They
can still hurt, which is §6.

This does not generalize to real serving, where prefix reuse is common and the
cache earns its memory.

---

## 6. Memory ceiling

### The OOM reproduces

With the default `--prompt-cache-size 10`, ISL 8,192 fails with a hard Metal
allocation error. Captured this time in `server-cache10.log` rather than
transcribed from a terminal:

```
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

The mechanism is prompt-cache retention, not model size. A single 8,192-token
request completes; ten retained caches do not. With `--prompt-cache-size 1`, ISL
8,192 completed 20 of 20 requests in this session (§3).

### What happened, in order

Reconstructed from `server-cache10.log` and `run_baseline.out`:

| Time | Event |
|---|---|
| 22:11:26 | `cache10_isl8192` profiling begins |
| — | two 8,192-token requests complete |
| ~22:12:12 | the third request triggers the Metal OOM, ~50 s in |
| 22:12 – 22:49 | AIPerf does not recover from the failed server generation thread |
| ~22:49:48 | interrupted manually |

So the OOM itself is fast, and the 38 minutes that follow are a **client hang
after the server's generation thread died**, not slow degradation under memory
pressure.

An earlier draft of this section claimed the run "degraded into something
unusable and fails slowly" over 40 minutes, based on a count of 320
prompt-processing lines in the server log. That count was wrong: roughly 300 of
those lines belong to the preceding 100-request ISL-512 run written to the same
log, and only about 20 belong to the long-context run. The claim is withdrawn.

What the evidence supports: the OOM is real and reproducible, and the client does
not recover from it. What it does **not** support is any statement about swap
changing the failure mode between rounds. The earlier round's faster death and
this one's client hang were never compared under controlled conditions.

This run is recorded as `cache10_isl8192` with no `profile_export_aiperf.json`,
because AIPerf was interrupted before writing one. The partial per-request
records and the server log are the evidence.

---

## 7. Repeatability

### 7.1 Fresh process

ISL 512 / concurrency 1, re-run 40 minutes later against a server restarted from
scratch:

| Run | TTFT p50 (ms) | ITL p50 (ms) |
|---|---:|---:|
| `isl512` | 553.34 | 27.02 |
| `isl512_freshproc` | 547.48 | 26.89 |

1.1% and 0.5% apart, across a process restart. The earlier round only tested
repeatability within one server process; this closes that gap.

### 7.2 Session drift is larger than fresh-process variance

ISL 512, concurrency 1, `--prompt-cache-size 1` — the same configuration
measured at five points in the session:

| Run | Time | n | ITL p50 (ms) |
|---|---|---:|---:|
| `warm1` | 21:08 | 10 | 22.81 |
| `warm2` | 21:09 | 10 | 22.87 |
| `warm3` | 21:10 | 10 | 23.26 |
| `isl512` | 21:16 | 100 | 27.02 |
| `isl512_freshproc` | 21:51 | 100 | 26.89 |

ITL ranges from 22.81 to 27.02 ms, about 18%, with no change in workload.
Thermal behaviour and accumulated memory pressure are the obvious candidates;
neither was instrumented.

Two caveats on this comparison. The three early points are 10-request medians
and the two later ones are 100-request medians, so sample size is confounded with
time; comparing the first ten seeded requests of each run would be a cleaner
test and is left undone. And `cache10_isl512` at 22:05 measured 24.31 ms, which
sits inside this range but is excluded here because its cache configuration
differs (§5.2).

**This is the most important caveat in the document.** An 18% drift at fixed
configuration is the same magnitude as several of the differences reported in
§3 and §5. Any single comparison smaller than roughly 20% should be treated as
unresolved unless the two runs were adjacent in time.

### 7.3 Warm-up is a process-start effect, not a server effect

Three identical 10-request runs at the start of the session:

| Run | TTFT p50 (ms) | ITL p50 (ms) |
|---|---:|---:|
| `warm1` | 532.42 | 22.81 |
| `warm2` | 531.01 | 22.87 |
| `warm3` | 534.18 | 23.26 |

Flat. The earlier round reported prompt throughput of 8.3 → 77.5 → 105.9 tok/s
across successive runs and described it as a 9–13× warm-up swing; those were
`mlx_lm.generate` CLI invocations, each a fresh process paying kernel
compilation. Against a running server the effect is absorbed inside the first
few requests and is not visible at run granularity.

The earlier claim was true of fresh processes and wrong about warm servers. A
20-request warm-up is still discarded here as a precaution.

---

## 8. Limitations

1. **Swap was active for most of the session.** `vm.swapusage` showed 0.00M used
   through `isl128` and `isl512`, then a 3 GB swap file appeared during
   `isl2048` and stayed, with 1.3–2.3 GB used for every subsequent run. The two
   short-ISL points are therefore swap-free and everything after them is not,
   which is an asymmetry sitting directly under the §3 curve. `vm.swapusage`
   counts pages written out at any time rather than pages being read back during
   a run, so this is evidence of memory pressure rather than proof of paging
   stalls. Page-in counters would settle it and were not captured.

2. **Co-located load client.** AIPerf runs on the same machine as the server and
   competes with it for CPU. This may shift absolute values, and it may
   contribute unevenly as concurrency rises, since the client's own work grows
   with the request rate. Its impact was not isolated, so the concurrency sweep
   in particular carries an unquantified client-side component.

3. **No GPU telemetry.** AIPerf reports `Platform: unknown` and collects no
   counters on Apple Silicon — its backends are DCGM and pynvml. No utilization,
   power or bandwidth data, so joules per token is not measurable here and every
   causal statement in §3 and §5 is interpretation rather than result.

4. **Output length is not pinned.** OSL ranges 122.60–127.95 across runs against
   a 128 cap. `mlx_lm.server` ignores AIPerf's `min_tokens` and `ignore_eos`,
   verified by passing them explicitly in the earlier round and observing no
   change. `max_tokens` caps but does not pin.

5. **Session drift of ~18% at fixed configuration.** See §7.2.

6. **ISL 8,192 is n=20**, not n=100, because each request takes ~18 s.

7. **One model, one precision.** Qwen3-0.6B at bf16. Nothing extrapolates to 7B+
   models, where weight streaming dominates differently.

8. **`cache10_isl8192` has no client-side export.** The deliberate-OOM run hung
   after the server's generation thread died and was interrupted manually 38
   minutes later, so AIPerf never wrote a summary (§6). The session script is
   therefore **not fully unattended**: it completes 14 of 15 runs on its own and
   needs a manual kill on the last one. It has no watchdog.

9. **The model revision is recorded, not pinned.** See §2. A future run of this
   script against a moved repo would serve different weights under the same
   commands.

10. **Runs are ordered, and order is confounded with ISL.** The ISL sweep runs
   from short to long, so context length and time-in-session increase together.
   Given §7.2 this cannot be separated from the present data. A shuffled or
   interleaved sweep would fix it.

---

## 9. Evidence

Every run in this document has a committed artifact directory:

```
results_parity/
    manifest.txt                versions, model SHA, swap per run, timings
    server-{main,restart,pc1,cache10}.log
    mlx_lm_server_help.txt
    aiperf-stdout.log
    <run>/                      one per configuration, 15 total
        inputs.json             rendered request payloads
        profile_export.jsonl    per-request records
        profile_export_aiperf.{csv,json}
        profile_export_console.txt
        logs/aiperf.log
```

`inputs.json` is what proves thinking suppression took effect.
`server-cache10.log` is what proves the OOM. Neither existed in the earlier
round.

Not captured: GPU counters (§8.3), page-in statistics (§8.1), and a client-side
export for `cache10_isl8192` (§8.8).

---

## 10. Open questions

Left to future work, listed so the gaps are explicit rather than implied.

1. A shuffled ISL sweep, to separate context length from session drift (§8.9).
2. Page-in counters alongside `vm.swapusage`, to establish whether the §3
   falloff at 8,192 is attention cost or paging (§8.1).
3. Per-request TTFT distributions from the committed `profile_export.jsonl`, to
   test the batching interpretation in §5.1 without new runs. This one needs no
   hardware.
4. Any future platform: `--random-seed`, a pinned model revision, an explicit
   thinking setting verified in the rendered payload, and `--artifact-dir` per
   configuration, with a single-request dry run before any sweep.
