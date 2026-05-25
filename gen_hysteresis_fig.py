"""Generate Fig. F0 — stable vs degrading hysteretic cycle comparison."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# ── Shared parameters ────────────────────────────────────────────────────────
COLORS_STABLE = ["#2166AC", "#4393C3", "#92C5DE"]   # blue shades
COLORS_DEGRAD = ["#D6604D", "#F4A582", "#FDDBC7", "#B2182B"]  # red shades

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Stable Coulomb friction (RFD)
# Perfect rectangle: all cycles identical, constant enclosed area
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

d_max = 1.0   # normalized max displacement
Mf    = 1.0   # normalized friction moment

n_cycles = 3
for idx in range(n_cycles):
    color = COLORS_STABLE[idx]
    lw = 1.8 - idx * 0.2
    # Rectangular Coulomb loop: explicit corners
    # Sequence: (-d,-Mf) → (-d,+Mf) → (+d,+Mf) → (+d,-Mf) → (-d,-Mf)
    d_rect = [-d_max, -d_max,  d_max,  d_max, -d_max]
    f_rect = [   -Mf,    +Mf,    +Mf,    -Mf,    -Mf]
    ax.plot(d_rect, f_rect, color=color, lw=lw,
            label=f"Cycle {idx+1}" if idx < 2 else "Cycles 1–3 (identical)")

# Shade the enclosed area on the first cycle
ax.fill([-d_max, -d_max, d_max, d_max],
        [-Mf,   +Mf,   +Mf,  -Mf],
        color="#2166AC", alpha=0.10)

# Reference lines
ax.axhline(0,  color="k", lw=0.6, ls="--", alpha=0.35)
ax.axvline(0,  color="k", lw=0.6, ls="--", alpha=0.35)
ax.axhline(+Mf, color="k", lw=0.5, ls=":", alpha=0.4)
ax.axhline(-Mf, color="k", lw=0.5, ls=":", alpha=0.4)

# Labels
ax.annotate("$+M_f$", xy=(0.02, +Mf + 0.04), xycoords=("axes fraction", "data"),
            fontsize=9, ha="left", color="k", alpha=0.6)
ax.annotate("$-M_f$", xy=(0.02, -Mf - 0.12), xycoords=("axes fraction", "data"),
            fontsize=9, ha="left", color="k", alpha=0.6)

ax.annotate("Constant area\n(uniform dissipation\nper cycle)",
            xy=(0.0, 0.0), xytext=(0.55, 0.35),
            xycoords="data", textcoords="axes fraction",
            fontsize=8.5, ha="center", color="#2166AC",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#2166AC", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="#2166AC",
                            connectionstyle="arc3,rad=0.2", lw=0.9))

ax.set_xlabel("Relative displacement (normalized)")
ax.set_ylabel("Restoring force (normalized)")
ax.set_title("(a) Stable hysteretic cycle\n(Coulomb friction — RFD)")
ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.55, 1.55)
ax.set_aspect("equal")

custom_lines = [Line2D([0], [0], color=c, lw=1.6) for c in COLORS_STABLE]
ax.legend(custom_lines, ["Cycle 1", "Cycle 2", "Cycle 3"],
          loc="upper left", framealpha=0.85)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Degrading / pinching cycle (typical of yielding members)
# Each successive cycle has reduced peak force and pinched shape
# ══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

n_cycles_d = 4
d_amp_base = 0.85
Fy_base    = 1.0

t = np.linspace(0, 2 * np.pi, 500)

for idx in range(n_cycles_d):
    color = COLORS_DEGRAD[idx]
    lw = 1.8 - idx * 0.15

    # Progressive degradation
    strength_factor = 1.0 - 0.18 * idx          # peak force degrades
    d_amp = d_amp_base * (1 + 0.12 * idx)       # displacement amplitude grows

    d_cycle = d_amp * np.sin(t)

    # Pinching model: tanh gives smooth saturation, but we reduce stiffness
    # near zero to simulate pinching (slack, gap closing)
    pinch = 0.25 + 0.05 * idx                   # pinching gets worse
    f_cycle = (Fy_base * strength_factor *
               np.tanh(d_cycle / (pinch * d_amp)) *
               (1 - 0.15 * np.abs(np.sin(t))))  # slight in-cycle softening

    ax2.plot(d_cycle, f_cycle, color=color, lw=lw)

# Shade first cycle area for reference
d0 = d_amp_base * np.sin(t)
f0 = Fy_base * np.tanh(d0 / (0.25 * d_amp_base)) * (1 - 0.15*np.abs(np.sin(t)))
ax2.fill(d0, f0, color="#D6604D", alpha=0.08)

ax2.axhline(0, color="k", lw=0.6, ls="--", alpha=0.35)
ax2.axvline(0, color="k", lw=0.6, ls="--", alpha=0.35)

ax2.annotate("Shrinking area\n(degrading dissipation\ncapacity per cycle)",
             xy=(-0.1, -0.2), xytext=(0.18, 0.22),
             xycoords="data", textcoords="axes fraction",
             fontsize=8.5, ha="center", color="#B2182B",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec="#B2182B", alpha=0.85),
             arrowprops=dict(arrowstyle="->", color="#B2182B",
                             connectionstyle="arc3,rad=-0.25", lw=0.9))

ax2.annotate("Pinching\n(reduced stiffness\nnear zero)", xy=(0.02, -0.04),
             xytext=(0.65, 0.18), xycoords="data", textcoords="axes fraction",
             fontsize=8.5, ha="center", color="#666666",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="gray", alpha=0.85),
             arrowprops=dict(arrowstyle="->", color="gray",
                             connectionstyle="arc3,rad=0.3", lw=0.9))

ax2.set_xlabel("Relative displacement (normalized)")
ax2.set_ylabel("Restoring force (normalized)")
ax2.set_title("(b) Unstable hysteretic cycle\n(degrading/pinching — yielding members)")
ax2.set_xlim(-1.55, 1.55)
ax2.set_ylim(-1.35, 1.35)

custom_lines2 = [Line2D([0], [0], color=c, lw=1.6) for c in COLORS_DEGRAD]
ax2.legend(custom_lines2, ["Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4"],
           loc="upper left", framealpha=0.85)

plt.tight_layout(pad=1.2)
out = (r"c:\Users\Tartolas\Desktop\INVESTIGACION FRANCIA\Proyecto"
       r"\latex_paper\figs\F0_hysteresis_comparison.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
