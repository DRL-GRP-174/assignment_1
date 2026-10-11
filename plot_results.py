"""Reads results/*/*.csv and makes the learning-curve + KL figures used in the slides."""
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"0.0": "#7F8C8D", "0.1": "#2E86C1", "0.2": "#C0392B", "0.3": "#D68910"}
LABEL = {"0.0": "no clipping", "0.1": "ε = 0.1", "0.2": "ε = 0.2", "0.3": "ε = 0.3"}


def load(folder):
    runs = {}
    for f in glob.glob(os.path.join(folder, "*.csv")):
        clip = os.path.basename(f).split("_clip")[1].split("_s")[0]
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        data = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
        runs.setdefault(clip, []).append(data)
    return runs


def smooth(x, k=5):
    if len(x) <= k:
        return x
    xp = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")   # edge-pad so the ends don't dip
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def curves(ax, runs, key, ylabel, band=True):
    for clip in sorted(runs):
        steps = runs[clip][0]["step"]
        ys = np.stack([smooth(r[key]) for r in runs[clip]])
        m, s = ys.mean(0), ys.std(0)
        ax.plot(steps, m, color=COLORS[clip], lw=2, label=LABEL[clip])
        if band:
            ax.fill_between(steps, m - s, m + s, color=COLORS[clip], alpha=0.15)
        else:  # log-scale plot: show each seed faintly instead of a +-std band
            for y in ys:
                ax.plot(steps, y, color=COLORS[clip], lw=0.6, alpha=0.35)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def summary(runs, last_frac=0.2):
    # like the paper: average over training AND final performance
    out = {}
    for clip, rs in runs.items():
        avg_all = [r["avg_return"].mean() for r in rs]
        n = max(1, int(len(rs[0]["avg_return"]) * last_frac))
        final = [r["avg_return"][-n:].mean() for r in rs]
        out[clip] = (np.mean(avg_all), np.std(avg_all), np.mean(final), np.std(final))
    return out


os.makedirs("figures", exist_ok=True)
default = load("results/default")
aggr = load("results/aggressive")

fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
curves(ax, default, "avg_return", "avg episode return")
ax.set_title("CartPole-v1, 3 seeds (mean ± std)", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/curves_default.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
curves(axes[0], aggr, "avg_return", "avg episode return")
axes[0].set_title("Aggressive updates (lr 1e-3, K = 20)", fontsize=10)
axes[0].legend(fontsize=8, loc="lower right")
curves(axes[1], aggr, "approx_kl", "approx KL(π_old || π)", band=False)
axes[1].set_yscale("log")
axes[1].set_ylim(1e-5, 10)
axes[1].set_title("How far each update moves the policy", fontsize=10)
plt.tight_layout()
plt.savefig("figures/aggressive.png")

print("\nExperiment A (default)    avg-over-training   final-20%")
for clip, (a, sa, f, sf) in sorted(summary(default).items()):
    print(f"  {LABEL[clip]:12s}  {a:7.1f} ± {sa:5.1f}     {f:7.1f} ± {sf:5.1f}")
print("\nExperiment B (aggressive)")
for clip, (a, sa, f, sf) in sorted(summary(aggr).items()):
    print(f"  {LABEL[clip]:12s}  {a:7.1f} ± {sa:5.1f}     {f:7.1f} ± {sf:5.1f}")
out = {}
for name, runs in [("default", default), ("aggressive", aggr)]:
    s = summary(runs)
    for clip in sorted(runs):
        kl = float(np.mean([r["approx_kl"].mean() for r in runs[clip]]))
        cf = float(np.mean([r["clip_frac"].mean() for r in runs[clip]]))
        first500 = [int(r["step"][np.argmax(r["avg_return"] >= 475)]) if (r["avg_return"] >= 475).any() else None
                    for r in runs[clip]]
        print(f"  [{name}] {LABEL[clip]:12s} mean approx KL {kl:.4f} | ratios outside ±0.2: {cf:.3f} | "
              f"first step avg>=475: {first500}")
        out.setdefault(name, {})[clip] = dict(avg=s[clip][0], avg_sd=s[clip][1], final=s[clip][2],
                                              final_sd=s[clip][3], kl=kl, clipfrac=cf, solved_at=first500)
import json
json.dump(out, open("figures/summary.json", "w"), indent=2)
