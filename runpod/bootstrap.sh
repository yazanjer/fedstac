#!/usr/bin/env bash
# FedStaP v0.5.0 — RunPod runner. Idempotent and resumable: re-running it (container restart,
# preemption, manual paste into the web terminal) continues where it stopped.
#
# Pod environment:
#   STAGES        space-separated stages, e.g. "smoke" | "sanity pilot" | "sanity full"   (required)
#   FEDSTAC_REF   branch or commit of yazanjer/fedstac to run (default v0.5.0)
#   GITHUB_TOKEN  {{ RUNPOD_SECRET_yazan_github }}  (write access; used only in git headers, never printed)
#   WORKERS       parallel runs (default: number of CPUs)
# Everything durable lives on the network volume at /workspace.
set -uo pipefail
WS=${WS:-/workspace}
REF=${FEDSTAC_REF:-v0.5.0}
REPO_URL=${REPO_URL:-https://github.com/yazanjer/fedstac.git}
OUT=$WS/outputs
mkdir -p "$WS" "$OUT"
LOG=$OUT/stage_log.txt
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
auth_hdr() { printf 'AUTHORIZATION: basic %s' "$(printf 'x-access-token:%s' "${GITHUB_TOKEN:-}" | base64 -w0)"; }

DONE_TAG=$(echo "$STAGES $REF" | tr ' /' '__')
if [ -f "$OUT/DONE_$DONE_TAG" ]; then
  log "stages '$STAGES' at $REF already complete; idling"; sleep infinity
fi

# ---------------------------------------------------------------- 1. system + python deps
log "bootstrap start: stages='$STAGES' ref=$REF cpus=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G"
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
export PIP_CACHE_DIR=$WS/pip-cache
if [ -z "${SKIP_PIP:-}" ]; then
python3 -m pip install -q --upgrade pip >/dev/null 2>&1
python3 -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
  || python3 -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install -q numpy==2.1.3 scipy==1.16.3 scikit-learn==1.6.1 pandas pyyaml \
  hydra-core==1.3.2 omegaconf==2.3.0 gdown matplotlib || { log "pip install failed"; exit 1; }
fi
python3 -m pip freeze > "$OUT/requirements.freeze.txt"
{ uname -a; python3 -V; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g; } > "$OUT/env.txt" 2>&1

# ---------------------------------------------------------------- 2. code
if [ ! -d "$WS/fedstac/.git" ]; then
  git clone -q "$REPO_URL" "$WS/fedstac" || { log "clone failed"; exit 1; }
fi
cd "$WS/fedstac"
git fetch -q origin "$REF" && git checkout -q -f FETCH_HEAD || { log "fetch $REF failed"; exit 1; }
log "code at $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- 3. data (hash-verified)
python3 - <<'PY' || { echo "data verification failed" | tee -a "$LOG"; exit 1; }
import hashlib, json, os, shutil, subprocess
from pathlib import Path
ws = Path(os.environ.get("WS", "/workspace"))
src = json.loads(Path("configs/data/prepared_sources.json").read_text())
for ds, s in src.items():
    d = ws / "data" / "prepared" / ds; d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f"configs/data/meta/{ds}.json", d / "meta.json")
    meta_sha = json.loads((d / "meta.json").read_text())["prepared_sha256"]
    assert meta_sha == s["sha256"], f"{ds}: meta.json and sources disagree"
    f = d / "prepared.npz"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for attempt in range(3):
        if f.exists() and sha(f) == meta_sha:
            break
        subprocess.run(["gdown", "--quiet", "--id", s["drive_id"], "-O", str(f)], check=False)
    got = sha(f) if f.exists() else None
    print(f"{ds}: sha256 {got} {'OK' if got == meta_sha else 'MISMATCH'}")
    assert got == meta_sha, f"{ds}: prepared.npz hash mismatch"
PY
log "datasets verified against meta.json prepared_sha256"

# ---------------------------------------------------------------- 4. results branch worktree
TREE=$WS/results-tree
if [ ! -d "$TREE/.git" ]; then
  git -c http.extraheader="$(auth_hdr)" clone -q --branch v0.5.0-results --single-branch "$REPO_URL" "$TREE" 2>/dev/null \
  || { git init -q "$TREE" && git -C "$TREE" remote add origin "$REPO_URL" && git -C "$TREE" checkout -q --orphan v0.5.0-results \
       && echo "# FedStaP v0.5.0 results (RunPod)" > "$TREE/README.md" && git -C "$TREE" add README.md \
       && git -C "$TREE" -c user.name="FedStaP RunPod runner" -c user.email=noreply@anthropic.com commit -q -m "init results branch"; }
fi
sync() { python3 scripts/sync_results.py --out "$OUT" --tree "$TREE" ${1:+--message "$1"} 2>&1 | tee -a "$LOG"; }
( while true; do sleep 900; sync >/dev/null 2>&1; done ) &
SYNC_PID=$!

# ---------------------------------------------------------------- 5. stages
WORKERS=${WORKERS:-$(nproc)}
for st in $STAGES; do
  log "stage $st: start (workers=$WORKERS)"
  python3 scripts/run_scaling.py --stage "$st" --workers "$WORKERS" --out "$OUT" --data "$WS/data/prepared" \
      >> "$OUT/run_$st.log" 2>&1
  rc=$?
  tail -n 3 "$OUT/run_$st.log" | tee -a "$LOG"
  sync "results v0.5.0: stage $st finished rc=$rc"
  if [ $rc -ne 0 ]; then
    log "stage $st failed (rc=$rc); later stages not started"
    kill $SYNC_PID 2>/dev/null; sleep infinity
  fi
  log "stage $st: done"
done
touch "$OUT/DONE_$DONE_TAG"
kill $SYNC_PID 2>/dev/null
sync "results v0.5.0: stages '$STAGES' complete"
log "all stages complete"

# ---------------------------------------------------------------- 6. stop the pod (billing)
if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1 && log "pod stop requested"
fi
sleep infinity
