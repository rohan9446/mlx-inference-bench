#!/usr/bin/env bash
#
# M1 final session. One command, unattended, ~60-75 minutes.
#
#   chmod +x run_baseline.sh
#   caffeinate -i ./run_baseline.sh 2>&1 | tee run_baseline.out
#
# Runs from the project root with .venv present. Everything lands in
# results_parity/ : one artifact directory per configuration, four server
# logs, and a manifest with versions, swap readings and a summary table.
#
# What this collects, and why:
#
#   A. Parity sweeps      the ISL and concurrency sweeps, re-run with
#                         thinking disabled, a fixed seed, explicit
#                         temperature and a separate artifact dir per run.
#   B. Warm-up curve      cold vs warm on identical requests, preserved
#                         this time instead of quoted from a terminal.
#   C. Fresh-process      one config repeated after a server restart, which
#      repeatability      the previous round never tested.
#   D. Prefill ablation   --prompt-concurrency 1 vs its default of 8, at
#                         fixed request concurrency. This is the experiment
#                         that decides whether prefill saturation is a
#                         capacity limit or a parallelism limit.
#   E. Cache ablation     default --prompt-cache-size 10 vs 1, including a
#                         deliberate OOM at ISL 8192 so the Metal traceback
#                         is captured as evidence rather than transcribed.
#
# Nothing here is destructive. E is expected to fail, on purpose, last.

set -uo pipefail

MODEL="Qwen/Qwen3-0.6B"
PORT=8080
SEED=42
OUT="results_parity"
EXTRA='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'

mkdir -p "$OUT"
MANIFEST="$OUT/manifest.txt"
: > "$MANIFEST"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MANIFEST"; }
swap() { echo "    swap: $(sysctl -n vm.swapusage)" | tee -a "$MANIFEST"; }

SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  sleep 2
}
trap stop_server EXIT

start_server() {            # start_server <logname> [extra flags...]
  local name=$1; shift
  log "server start [$name]: $*"
  mlx_lm.server --model "$MODEL" --port "$PORT" \
    --chat-template-args '{"enable_thinking": false}' \
    "$@" > "$OUT/server-$name.log" 2>&1 &
  SERVER_PID=$!
  local i
  for i in $(seq 1 90); do
    curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FATAL: server died on startup, see $OUT/server-$name.log"; exit 1
    fi
    sleep 2
  done
  log "  up (pid $SERVER_PID)"
}

run () {                    # run <name> <isl> <conc> <count>
  local name=$1 isl=$2 conc=$3 count=$4
  log "run $name  ISL=$isl conc=$conc n=$count"
  aiperf profile --model "$MODEL" --endpoint-type chat --streaming \
    --url "http://localhost:$PORT" \
    --concurrency "$conc" \
    --synthetic-input-tokens-mean "$isl" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 --output-tokens-stddev 0 \
    --extra-inputs "$EXTRA" \
    --random-seed "$SEED" \
    --request-count "$count" \
    --artifact-dir "$OUT/$name" \
    >> "$OUT/aiperf-stdout.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && log "  aiperf exited $rc (recorded, continuing)"
  swap
}

# ============================================================ 0. preflight

log "M1 final session"
log "host: $(uname -srm)  macOS $(sw_vers -productVersion)"
source .venv/bin/activate
log "python: $(python -V 2>&1)"
for p in mlx mlx-lm mlx-metal transformers numpy aiperf; do
  log "  $p: $(pip show "$p" 2>/dev/null | awk '/^Version/{print $2}')"
done
log "seed: $SEED"
log "extra-inputs: $EXTRA"

REV=$(python - <<'PY'
import urllib.request, json
u = "https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/main"
try: print(json.load(urllib.request.urlopen(u, timeout=20))["sha"])
except Exception as e: print(f"unresolved ({e})")
PY
)
log "model: $MODEL @ $REV  (resolved at run time; mlx_lm has no revision flag)"
mlx_lm.server --help > "$OUT/mlx_lm_server_help.txt" 2>&1 || true
log "server defaults: prompt-cache-size 10, prompt-concurrency 8, decode-concurrency 32"
swap

# ====================================== A/B. main server: cache 1, defaults

start_server main --prompt-cache-size 1

# B. warm-up curve: three identical small runs, cold to warm, preserved.
log "--- warm-up curve ---"
run warm1 512 1 10
run warm2 512 1 10
run warm3 512 1 10

# preflight guard: did thinking suppression actually reach the request?
log "--- preflight probe ---"
run probe 512 1 5

python - "$OUT" <<'PY' | tee -a "$MANIFEST"
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
inp = next(out.glob("probe/**/inputs.json"), None)
if inp is None:
    print("PREFLIGHT FAIL: no inputs.json written"); sys.exit(2)
txt = inp.read_text()
ok_think = "enable_thinking" in txt
ok_temp = '"temperature"' in txt
print(f"preflight: enable_thinking in payload = {ok_think}")
print(f"preflight: temperature in payload     = {ok_temp}")
if not ok_think:
    print("PREFLIGHT FAIL: chat_template_kwargs did not reach the request.")
    print("  Nothing further was run. Fix --extra-inputs and retry.")
    sys.exit(2)
js = next(out.glob("probe/**/profile_export_aiperf.json"), None)
if js:
    d = json.loads(js.read_text())
    osl = d.get("output_sequence_length") or d.get("output_token_count")
    if isinstance(osl, dict):
        print(f"preflight: OSL avg = {osl.get('avg')}  min = {osl.get('min')}")
        if (osl.get("avg") or 0) < 100:
            print("  NOTE: output length is well below 128. Thinking is off and")
            print("  the model reaches EOS early. Sweeps continue; ITL stays")
            print("  valid per token but is measured over fewer tokens.")
PY
[ ${PIPESTATUS[0]} -eq 2 ] && { log "aborted at preflight"; exit 2; }
log "preflight passed"

# A. ISL sweep, concurrency 1
log "--- ISL sweep ---"
run isl128   128  1 100
run isl512   512  1 100
run isl2048  2048 1 100
run isl8192  8192 1 20

# A. concurrency sweep, ISL 512, server defaults (prompt-concurrency 8)
log "--- concurrency sweep (prompt-concurrency 8, the default) ---"
run conc2 512 2 100
run conc4 512 4 100
run conc8 512 8 100

stop_server

# ================================= C. fresh-process repeatability, same cfg

log "--- fresh-process repeatability ---"
start_server restart --prompt-cache-size 1
run isl512_freshproc 512 1 100
stop_server

# ============================== D. prefill parallelism ablation (pc = 1)

log "--- prefill ablation: --prompt-concurrency 1 vs default 8 ---"
start_server pc1 --prompt-cache-size 1 --prompt-concurrency 1
run pc1_conc4 512 4 100
run pc1_conc8 512 8 100
stop_server

# ===================== E. cache ablation + deliberate OOM (expected to fail)

log "--- cache ablation: default --prompt-cache-size 10 ---"
start_server cache10
run cache10_isl512 512 1 100
log "next run is EXPECTED to fail with a Metal OOM. That is the point."
run cache10_isl8192 8192 1 20
grep -h "Prompt Cache:" "$OUT/server-cache10.log" | sort -u | tail -3 \
  | tee -a "$MANIFEST" || true
if grep -qi "insufficient memory\|OutOfMemory" "$OUT/server-cache10.log"; then
  log "OOM reproduced and captured in server-cache10.log"
else
  log "NOTE: no OOM this time. Worth reporting as such."
fi
stop_server

# =================================================================== report

log "--- done ---"
swap

python - "$OUT" <<'PY' | tee -a "$MANIFEST"
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
names = ["warm1","warm2","warm3","isl128","isl512","isl2048","isl8192",
         "conc2","conc4","conc8","isl512_freshproc","pc1_conc4","pc1_conc8",
         "cache10_isl512","cache10_isl8192"]
print(f"\n{'run':<18}{'TTFT p50':>10}{'ITL p50':>10}{'prefill':>10}"
      f"{'OSL avg':>9}{'err%':>7}")
for n in names:
    js = next(out.glob(f"{n}/**/profile_export_aiperf.json"), None)
    if not js:
        print(f"{n:<18}{'(no export)':>46}"); continue
    d = json.loads(js.read_text())
    g = lambda k, s="p50": (d.get(k) or {}).get(s)
    ttft, itl = g("time_to_first_token"), g("inter_token_latency")
    isl = g("input_sequence_length","avg")
    pre = (isl/ttft*1000) if (ttft and isl) else None
    f = lambda v, p=2: "n/a" if v is None else f"{v:,.{p}f}"
    print(f"{n:<18}{f(ttft):>10}{f(itl):>10}{f(pre,0):>10}"
          f"{f(g('output_sequence_length','avg')):>9}"
          f"{f((d.get('request_error_rate') or {}).get('avg')):>7}")
print("\nprefill = input_sequence_length / TTFT, tokens/s (user-visible rate)")
PY

log "results in $OUT/  -- send manifest.txt"
