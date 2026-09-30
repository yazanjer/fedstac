"""Copy durable results (no model weights) from an output root into a git worktree and push.

The worktree is a checkout of branch v0.5.0-results. Authentication comes from $GITHUB_TOKEN via an
HTTP header passed on the command line of git only; the token is never written to disk or printed.
Usage: python scripts/sync_results.py --out /workspace/outputs --tree /workspace/results-tree
"""
import argparse
import base64
import os
import shutil
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--tree", required=True)
ap.add_argument("--branch", default="v0.5.0-results")
ap.add_argument("--message", default=None)
a = ap.parse_args()
out, tree = Path(a.out), Path(a.tree)
tok = os.environ.get("GITHUB_TOKEN", "")
hdr = ["-c", "http.extraheader=AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{tok}".encode()).decode()] if tok else []


def git(*args, check=True):
    r = subprocess.run(["git", *hdr, *args], cwd=tree, capture_output=True, text=True)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout).replace(tok, "***") if tok else (r.stderr or r.stdout)
        raise SystemExit(f"git {args[0]} failed: {msg.strip()[-400:]}")
    return r


dst = tree / "v050"
(dst / "runs").mkdir(parents=True, exist_ok=True)
n_new = 0
for f in (out / "runs").glob("*/results.json"):
    t = dst / "runs" / f"{f.parent.name}.json"
    if not t.exists() or t.stat().st_size != f.stat().st_size:
        shutil.copy2(f, t); n_new += 1
if (out / "index").exists():
    shutil.copytree(out / "index", dst / "index", dirs_exist_ok=True)
for name in ("stage_log.txt", "env.txt", "requirements.freeze.txt"):
    if (out / name).exists():
        shutil.copy2(out / name, dst / name)
git("add", "-A", "v050")
if git("diff", "--cached", "--quiet", check=False).returncode == 0:
    print("[sync] nothing new"); raise SystemExit(0)
git("-c", "user.name=FedStaP RunPod runner", "-c", "user.email=noreply@anthropic.com", "commit", "-q", "-m",
    a.message or f"results v0.5.0: {len(list((dst / 'runs').glob('*.json')))} runs ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})")
for attempt in range(3):
    if git("push", "-q", "origin", f"HEAD:{a.branch}", check=False).returncode == 0:
        print(f"[sync] pushed ({n_new} new/updated run files)"); break
    git("pull", "-q", "--rebase", "origin", a.branch, check=False)
    time.sleep(5)
else:
    print("[sync] push failed after 3 attempts (results remain on the volume)")
