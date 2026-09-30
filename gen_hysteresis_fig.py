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
ks    = 0.22  # elastic frame stiffness (used only in right panel)
w     = 0.13  # corner rounding width (used only in right panel)

def coulomb_parallelogram(d_max, Mf, ks, w, n=400):
    """
    Coulomb friction parallelogram cycle with elastic frame stiffness ks.
    Forward branch:  F = +Mf + ks*d
    Backward branch: F = -Mf + ks*d
    Transition centers shifted 2w inside the path so corners are never clipped.
    """
    d_f = np.linspace(-d_max, d_max, n)
    f_target_f = Mf + ks * d_f
    f_prev_f   = -Mf + ks * (-d_max)
    step_f = 0.5 * (1 + np.tanh((d_f + d_max - 2*w) / w))
    f_f = f_prev_f * (1 - step_f) + f_target_f * step_f

    d_b = np.linspace(d_max, -d_max, n)
    f_target_b = -Mf + ks * d_b
    f_prev_b   = Mf + ks * d_max
    step_b = 0.5 * (1 + np.tanh((d_b - d_max + 2*w) / (-w)))
    f_b = f_prev_b * (1 - step_b) + f_target_b * step_b

    d_cycle = np.concatenate([d_f, d_b[1:]])
    f_cycle = np.concatenate([f_f, f_b[1:]])
    return d_cycle, f_cycle

n_cycles = 3
for idx in range(n_cycles):
    color = COLORS_STABLE[idx]
    lw = 1.8 - idx * 0.2
    # Perfect rectangle: explicit corners, all cycles identical
    d_rect = [-d_max, -d_max,  d_max,  d_max, -d_max]
    f_rect = [   -Mf,    +Mf,    +Mf,    -Mf,    -Mf]
    ax.plot(d_rect, f_rect, color=color, lw=lw)

# Shade enclosed area
ax.fill([-d_max, -d_max, d_max, d_max],
        [-Mf,    +Mf,    +Mf,   -Mf],
        color="#2166AC", alpha=0.10)

# Reference lines
ax.axhline(0,  color="k", lw=0.6, ls="--", alpha=0.35)
ax.axvline(0,  color="k", lw=0.6, ls="--", alpha=0.35)
ax.axhline(+Mf, color="k", lw=0.5, ls=":", alpha=0.4)
ax.axhline(-Mf, color="k", lw=0.5, ls=":", alpha=0.4)

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


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Degrading cycle: nested parallelograms that shrink each cycle
# Same parallelogram shape as left panel; Mf decreases per cycle
# ══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

# Four nested cycles: Mf shrinks, displacement fixed
Mf_cycles = [1.00, 0.68, 0.40, 0.18]
labels_d   = ["Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4"]

# Draw filled bands from outermost to innermost so inner ones are on top
for idx in reversed(range(len(Mf_cycles))):
    Mf_d  = Mf_cycles[idx]
    color = COLORS_DEGRAD[idx]
    d_c, f_c = coulomb_parallelogram(d_max, Mf_d, ks, w)
    ax2.fill(d_c, f_c, color=color, alpha=0.13 + 0.04 * (len(Mf_cycles) - idx))

for idx, Mf_d in enumerate(Mf_cycles):
    color = COLORS_DEGRAD[idx]
    lw = 1.9 - idx * 0.15
    d_c, f_c = coulomb_parallelogram(d_max, Mf_d, ks, w)
    ax2.plot(d_c, f_c, color=color, lw=lw, label=labels_d[idx])

ax2.axhline(0, color="k", lw=0.6, ls="--", alpha=0.35)
ax2.axvline(0, color="k", lw=0.6, ls="--", alpha=0.35)

# Arrow showing shrinking direction (outer → inner)
ax2.annotate("", xy=(0.28, 0.56), xytext=(0.48, 0.70),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color="#B2182B",
                             lw=1.3, mutation_scale=10))
ax2.text(0.50, 0.71, "Shrinking area\n(degrading dissipation\nper cycle)",
         transform=ax2.transAxes, fontsize=8.5, color="#B2182B",
         va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B2182B", alpha=0.85))

ax2.set_xlabel("Relative displacement (normalized)")
ax2.set_ylabel("Restoring force (normalized)")
ax2.set_title("(b) Degrading hysteretic cycle\n(strength loss — uncontrolled frame)")
ax2.set_xlim(-1.55, 1.55)
ax2.set_ylim(-1.55, 1.55)
ax2.set_aspect("equal")

plt.tight_layout(pad=1.2)
out = (r"c:\Users\Tartolas\Desktop\INVESTIGACION FRANCIA\Proyecto"
       r"\latex_paper\figs\F0_hysteresis_comparison.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
