"""Re-draws Figure 1 of the PPO paper: one term of L^CLIP as a function of the ratio r,
for a positive and a negative advantage. We used this to convince ourselves of
*why* the min() makes the objective a pessimistic bound."""
import matplotlib.pyplot as plt
import numpy as np

eps = 0.2
r = np.linspace(0.0, 2.0, 400)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
for ax, A, title in [(axes[0], 1.0, "Advantage A > 0"), (axes[1], -1.0, "Advantage A < 0")]:
    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    lclip = np.minimum(unclipped, clipped)
    ax.plot(r, unclipped, "--", color="#9AA5B1", lw=1.5, label="r·A  (no clip)")
    ax.plot(r, lclip, color="#C0392B", lw=2.5, label="L_CLIP = min(r·A, clip(r)·A)")
    ax.axvline(1, color="#555", lw=0.8, ls=":")
    ax.axvspan(1 - eps, 1 + eps, color="#F5B7B1", alpha=0.35, label="trust zone [1-ε, 1+ε]")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("probability ratio  r = π_θ / π_θold")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("objective term")
axes[0].legend(fontsize=7.5, loc="upper left")
plt.tight_layout()
plt.savefig("figures/clip_objective.png")
print("saved figures/clip_objective.png")
