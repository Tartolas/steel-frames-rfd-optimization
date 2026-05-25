"""Generate Fig. LHS — Latin Hypercube Sampling design illustration.

Left  : conceptual 2-D comparison LHS vs. simple random sampling (n=12)
Right : lower-triangle pair plot of the 1,000 actual structures
        (variables: T1, masa_promedio, n_pisos), coloured by n_pisos.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import qmc
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

# ── Load real LHS samples ────────────────────────────────────────────────────
df = pd.read_parquet(
    r"c:\Users\Tartolas\Desktop\INVESTIGACION FRANCIA\Proyecto"
    r"\out_fase1\plan_estructuras.parquet"
)

T1   = df["T1_objetivo"].values
masa = df["masa_promedio_ton"].values
np_v = df["n_pisos"].values

N_PISOS_VALUES = [3, 5, 7, 10, 12, 15, 18, 20]
cmap_d = plt.get_cmap("viridis", len(N_PISOS_VALUES))
bounds = [2, 4, 6, 8.5, 11, 13.5, 16.5, 19, 21]
norm   = BoundaryNorm(bounds, cmap_d.N)

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.5))
gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.65],
                            wspace=0.38)

# ════════════════════════════════════════════════════════════════════════════
# LEFT — Conceptual panel: LHS vs. simple random (n=12, 2-D)
# ════════════════════════════════════════════════════════════════════════════
ax_concept = fig.add_subplot(gs_main[0])

n_concept = 12
rng = np.random.default_rng(42)

# Simple random sample
x_rand = rng.random(n_concept)
y_rand = rng.random(n_concept)

# LHS sample (scrambled)
sampler = qmc.LatinHypercube(d=2, seed=42, scramble=True)
lhs_pts = sampler.random(n=n_concept)
x_lhs, y_lhs = lhs_pts[:, 0], lhs_pts[:, 1]

# Grid lines (strata boundaries)
ticks = np.linspace(0, 1, n_concept + 1)
for t in ticks:
    ax_concept.axhline(t, color="gray", lw=0.45, ls="-", alpha=0.45)
    ax_concept.axvline(t, color="gray", lw=0.45, ls="-", alpha=0.45)

# Shade one full row and one full column to illustrate "exactly one per stratum"
stripe_alpha = 0.10
row_idx = 6    # highlight stratum row 6
col_idx = 2    # highlight stratum col 2
ax_concept.axhspan(ticks[row_idx], ticks[row_idx + 1],
                   color="#2166AC", alpha=stripe_alpha)
ax_concept.axvspan(ticks[col_idx], ticks[col_idx + 1],
                   color="#2166AC", alpha=stripe_alpha)

# Points
ax_concept.scatter(x_rand, y_rand, c="#D6604D", s=38, zorder=5,
                   marker="s", label="Simple random (SRS)", edgecolors="white", lw=0.4)
ax_concept.scatter(x_lhs, y_lhs, c="#2166AC", s=38, zorder=5,
                   marker="o", label="LHS (randomized)", edgecolors="white", lw=0.4)

# Mark the single LHS point that falls in the highlighted stratum
# Find it: row_idx stratum is [ticks[row_idx], ticks[row_idx+1]]
in_row = np.where((y_lhs >= ticks[row_idx]) & (y_lhs < ticks[row_idx + 1]))[0]
if len(in_row):
    ax_concept.scatter(x_lhs[in_row], y_lhs[in_row],
                       c="#2166AC", s=80, zorder=6,
                       marker="o", edgecolors="#1a3f6f", lw=1.2)

# Annotation: exactly one per stratum
ax_concept.annotate(
    "Exactly one\nsample per\nstratum",
    xy=(x_lhs[in_row[0]] if len(in_row) else 0.2,
        y_lhs[in_row[0]] if len(in_row) else 0.55),
    xytext=(0.72, 0.73),
    fontsize=8, ha="center", color="#2166AC",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2166AC", alpha=0.9),
    arrowprops=dict(arrowstyle="->", color="#2166AC",
                    connectionstyle="arc3,rad=-0.25", lw=0.9),
)

ax_concept.set_xlim(0, 1)
ax_concept.set_ylim(0, 1)
ax_concept.set_xlabel("Factor 1 (normalized)")
ax_concept.set_ylabel("Factor 2 (normalized)")
ax_concept.set_title("(a) LHS principle\n(n = 12 strata per axis)")
ax_concept.set_xticks([0, 0.5, 1])
ax_concept.set_yticks([0, 0.5, 1])
ax_concept.legend(loc="lower right", framealpha=0.9)

# ════════════════════════════════════════════════════════════════════════════
# RIGHT — Lower-triangle pair plot of real 1,000 structures
#         Variables: T1, masa_promedio, n_pisos
# ════════════════════════════════════════════════════════════════════════════
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 3, subplot_spec=gs_main[1], hspace=0.12, wspace=0.12
)

labels   = [r"$T_1$ (s)", r"$\bar{m}$ (t)", r"$n_p$ (stories)"]
data_arr = [T1, masa, np_v.astype(float)]
lims     = [(0.25, 3.05), (140, 360), (2, 21)]
n_vars   = 3

axes_grid = {}
for row in range(n_vars):
    for col in range(n_vars):
        if col > row:
            continue  # upper triangle — leave empty
        ax = fig.add_subplot(gs_right[row, col])
        axes_grid[(row, col)] = ax

        if row == col:
            # Diagonal: histogram of marginal (shows uniform coverage)
            ax.hist(data_arr[row], bins=20, color="#2166AC",
                    alpha=0.75, edgecolor="white", lw=0.4)
            ax.set_xlim(lims[row])
            ax.set_yticks([])
        else:
            # Off-diagonal scatter coloured by n_pisos
            sc = ax.scatter(data_arr[col], data_arr[row],
                            c=np_v, cmap=cmap_d, norm=norm,
                            s=3, alpha=0.55, lw=0)
            ax.set_xlim(lims[col])
            ax.set_ylim(lims[row])

        # Axis labels only on edges
        if row == n_vars - 1:
            ax.set_xlabel(labels[col], labelpad=3)
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(labels[row], labelpad=3)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis="both", which="both", length=2.5, labelsize=7.5)

# Title above the pair plot
axes_grid[(0, 0)].set_title("(b) Coverage of the 1,000 LHS structures",
                             loc="left", fontsize=10, pad=8)

# Colorbar for n_pisos
sm = ScalarMappable(cmap=cmap_d, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.96, 0.12, 0.013, 0.76])
cb = fig.colorbar(sm, cax=cbar_ax, ticks=N_PISOS_VALUES)
cb.set_label(r"$n_p$ (stories)", fontsize=9)
cb.ax.tick_params(labelsize=8)

# ── Save ─────────────────────────────────────────────────────────────────────
out = (r"c:\Users\Tartolas\Desktop\INVESTIGACION FRANCIA\Proyecto"
       r"\latex_paper\figs\F_LHS_design.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
