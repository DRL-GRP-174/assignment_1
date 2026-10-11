"""Runs the clip-range ablation (our small-scale version of Table 1 in the paper).

Experiment A: default hyper-params, eps in {none, 0.1, 0.2, 0.3}, 3 seeds each
Experiment B: "aggressive" updates (lr 1e-3, 20 epochs) - clip 0.2 vs no clip,
              to see what happens when the policy is pushed far from pi_old
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
SEEDS = [1, 2, 3]

jobs = []
for clip in [0.0, 0.1, 0.2, 0.3]:
    for s in SEEDS:
        jobs.append([PY, "ppo.py", "--clip", str(clip), "--seed", str(s), "--out", "results/default"])
for clip in [0.0, 0.2]:
    for s in SEEDS:
        jobs.append([PY, "ppo.py", "--clip", str(clip), "--seed", str(s),
                     "--lr", "1e-3", "--epochs", "20", "--out", "results/aggressive"])


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:]
    print(last, flush=True)


with ThreadPoolExecutor(max_workers=6) as ex:
    list(ex.map(run, jobs))
