#!/usr/bin/env python3
"""
Plots for the M1 characterization.

Data from results_parity/, session of 2026-09-19, thinking disabled,
seed 42, --prompt-cache-size 1 unless noted. See results.md.

Usage:  python plot_results.py
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
})

pow2 = FuncFormatter(lambda v, _: f"{int(v):,}")

# ---------------------------------------------------------------- data

isl = [128, 512, 2048, 8192]
ttft_ms = [282.19, 553.34, 2070.65, 12127.51]
itl_ms = [22.19, 27.02, 27.19, 47.91]
prefill = [454, 925, 989, 675]

conc = [1, 2, 4, 8]
c_ttft = [553.34, 990.06, 1647.63, 3316.97]
c_itl = [27.02, 47.19, 54.88, 61.35]

# prefill ablation: --prompt-concurrency 8 (default) vs 1
abl_conc = [4, 8]
abl_ttft_pc8 = [1647.63, 3316.97]
abl_ttft_pc1 = [1587.19, 1662.65]
abl_itl_pc8 = [54.88, 61.35]
abl_itl_pc1 = [57.92, 85.09]

# session drift: ISL 512, conc 1, --prompt-cache-size 1, identical config.
# cache10_isl512 (24.31 ms at 22:05) is excluded — different cache setting.
drift_label = ["warm1\n21:08\nn=10", "warm2\n21:09\nn=10", "warm3\n21:10\nn=10",
               "isl512\n21:16\nn=100", "fresh proc\n21:51\nn=100"]
drift_itl = [22.81, 22.87, 23.26, 27.02, 26.89]


# ---------------------------------------------------------------- fig 1
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(isl, itl_ms, marker="o", color="C0")
for x, y in zip(isl, itl_ms):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=10)
ax.axhspan(22.81, 27.02, color="C7", alpha=0.15, zorder=0)
ax.annotate("drift band at fixed ISL 512\n(22.8-27.0 ms, see results.md 7.2)",
            xy=(600, 25), fontsize=9, color="0.35", va="center")
ax.set_xscale("log", base=2)
ax.set_xticks(isl); ax.xaxis.set_major_formatter(pow2)
ax.set_xlabel("Input sequence length (tokens)")
ax.set_ylabel("Inter-token latency, p50 (ms)")
ax.set_title("Decode latency vs. context length\n"
             "Qwen3-0.6B bf16, MLX-LM, M1 8GB, concurrency 1", fontsize=11)
ax.set_ylim(0, 55)
fig.savefig("fig_itl_vs_context.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(isl, prefill, marker="s", color="C1")
for x, y in zip(isl, prefill):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=10)
ax.set_xscale("log", base=2)
ax.set_xticks(isl); ax.xaxis.set_major_formatter(pow2)
ax.set_xlabel("Input sequence length (tokens)")
ax.set_ylabel("Prefill rate, ISL / TTFT (tokens/s)")
ax.set_title("Prefill rate vs. context length\n"
             "Qwen3-0.6B bf16, MLX-LM, M1 8GB, concurrency 1", fontsize=11)
ax.set_ylim(0, 1150)
fig.savefig("fig_prefill_rate.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ax1.plot(conc, [v / c_ttft[0] for v in c_ttft], marker="o", color="C1",
         label="TTFT (prefill)")
ax1.plot(conc, [v / c_itl[0] for v in c_itl], marker="s", color="C0",
         label="ITL (decode)")
ax1.set_xscale("log", base=2); ax1.set_xticks(conc)
ax1.xaxis.set_major_formatter(pow2)
ax1.set_xlabel("Concurrent requests")
ax1.set_ylabel("Latency relative to concurrency 1")
ax1.set_title("Latency scaling", fontsize=11)
ax1.set_ylim(0, 7); ax1.legend()

ax2.plot(conc, c_ttft, marker="o", color="C1", label="TTFT p50")
ax2.plot(conc, c_itl, marker="s", color="C0", label="ITL p50")
ax2.set_xscale("log", base=2); ax2.set_yscale("log")
ax2.set_xticks(conc); ax2.xaxis.set_major_formatter(pow2)
ax2.set_xlabel("Concurrent requests")
ax2.set_ylabel("milliseconds (log scale)")
ax2.set_title("Absolute latency", fontsize=11)
ax2.legend()

fig.suptitle("Concurrency sweep: Qwen3-0.6B bf16, MLX-LM, M1 8GB, "
             "ISL 512 / OSL 128", fontsize=11)
fig.tight_layout()
fig.savefig("fig_concurrency.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
w, x = 0.35, range(len(abl_conc))
x8 = [i - w / 2 for i in x]
x1 = [i + w / 2 for i in x]

ax1.bar(x8, abl_ttft_pc8, w, color="C1", label="prompt-concurrency 8 (default)")
ax1.bar(x1, abl_ttft_pc1, w, color="C0", label="prompt-concurrency 1")
for xs, vs in ((x8, abl_ttft_pc8), (x1, abl_ttft_pc1)):
    for xi, v in zip(xs, vs):
        ax1.annotate(f"{v:,.0f}", (xi, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9)
ax1.set_xticks(list(x)); ax1.set_xticklabels([f"concurrency {c}" for c in abl_conc])
ax1.set_ylabel("TTFT p50 (ms)"); ax1.set_title("Time to first token", fontsize=11)
ax1.set_ylim(0, 3900); ax1.legend(fontsize=9)

ax2.bar(x8, abl_itl_pc8, w, color="C1")
ax2.bar(x1, abl_itl_pc1, w, color="C0")
for xs, vs in ((x8, abl_itl_pc8), (x1, abl_itl_pc1)):
    for xi, v in zip(xs, vs):
        ax2.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9)
ax2.set_xticks(list(x)); ax2.set_xticklabels([f"concurrency {c}" for c in abl_conc])
ax2.set_ylabel("ITL p50 (ms)"); ax2.set_title("Inter-token latency", fontsize=11)
ax2.set_ylim(0, 100)

fig.suptitle("Prefill batching trades TTFT for ITL (ISL 512, M1 8GB)",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig_prefill_ablation.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 5
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(range(len(drift_itl)), drift_itl, marker="o", color="C3")
ax.set_xticks(range(len(drift_label))); ax.set_xticklabels(drift_label, fontsize=9)
ax.set_ylabel("ITL p50 (ms)")
ax.set_title("Same workload, same machine, one session\n"
             "ISL 512, concurrency 1, cache size 1 — 18% spread", fontsize=11)
ax.set_ylim(0, 32)
ax.annotate("sample size differs between the early and late points;\n"
            "see results.md 7.2", xy=(0.02, 0.06), xycoords="axes fraction",
            fontsize=8.5, color="0.4")
fig.savefig("fig_session_drift.png"); plt.close(fig)

print("wrote fig_itl_vs_context.png, fig_prefill_rate.png, fig_concurrency.png,"
      " fig_prefill_ablation.png, fig_session_drift.png")
