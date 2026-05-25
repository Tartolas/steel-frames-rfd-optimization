"""
================================================================================
FIGURAS PARA EL PAPER — Optimización ML de RFD en marcos de acero
================================================================================
Genera figuras publication-quality que complementan los resultados del pipeline ML.

Figuras producidas (en out_ml/paper_figs/):
  F1  Comparación RMSE_CV: 4 modelos × 3 targets
  F2  Predicción vs valor real (OOF cross-validation, XGBoost, CR2_median)
  F3  Análisis de residuos del mejor modelo
  F4  Importancia de features — los 3 targets en paralelo
  F5  CR2 vs densidad de amortiguadores (N/n_pisos) — rendimiento decreciente
  F6  Mejora marginal por amortiguador adicional (normalizada por altura)
  F7  Curvas within-structure: CR2_best normalizado vs fracción de pisos cubiertos
  F8  Mapa de calor interacción N_Dampers × Mf → CR2_median
  F9  Análisis de convergencia extendido (todos los targets, muestra completa)

Uso:
    python 05_paper_figures.py

Requiere:
    - out_ml/dataset_ml.parquet
    - out_ml/models/XGBoost_median.pkl  (y demás modelos .pkl)
================================================================================
"""

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path("./out_ml")
MODELS_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "paper_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Estilo global ──────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

TARGETS = {
    "CR2_median": "Median (typical scenario)",
    "CR2_p10":    "P10 (favorable case)",
    "CR2_p90":    "P90 (conservative design)",
}
TARGET_COLORS = {
    "CR2_median": "#2166AC",
    "CR2_p10":    "#4DAC26",
    "CR2_p90":    "#D6604D",
}
MODEL_ORDER = ["LinearRegression", "Ridge", "RandomForest", "XGBoost"]
MODEL_COLORS = ["#AAAAAA", "#CCCCCC", "#74ADD1", "#F46D43"]

HEIGHT_GROUPS = {
    "Low-rise (3–5)":   (3, 5),
    "Mid-rise (6–10)":  (6, 10),
    "High-rise (11–15)": (11, 15),
    "Tall (16–20)":     (16, 20),
}
HEIGHT_COLORS = ["#1A9850", "#91CF60", "#FEE08B", "#D73027"]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Carga dataset ML y define feature columns."""
    agg = pd.read_parquet(OUT_DIR / "dataset_ml.parquet")

    feature_cols = (
        ["n_pisos_first", "T1_real_first", "Mf_KN_first", "N_Dampers_first",
         "densidad_dampers", "posicion_promedio_dampers", "posicion_normalizada",
         "concentracion_dampers", "energia_total_normalizada"]
        + [c for c in agg.columns if c.startswith("damper_piso")]
    )
    return agg, feature_cols


def load_model(name: str, suffix: str):
    path = MODELS_DIR / f"{name}_{suffix}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_oof_predictions(agg, feature_cols, target_col="CR2_median", n_splits=5):
    """Genera predicciones out-of-fold con XGBoost (o RF si XGB no está)."""
    X = agg[feature_cols].values
    y = agg[target_col].values
    groups = agg["estructura_id"].values

    suffix = target_col.replace("CR2_", "")
    model_loaded = load_model("XGBoost", suffix) or load_model("RandomForest", suffix)
    if model_loaded is None:
        return None, None

    # Determina el tipo de modelo y reconstruye con mismos hiperparámetros
    try:
        import xgboost as xgb
        estimator = xgb.XGBRegressor(n_estimators=100, max_depth=6,
                                     learning_rate=0.1, random_state=42,
                                     n_jobs=-1, verbosity=0)
        model_name = "XGBoost"
    except ImportError:
        estimator = RandomForestRegressor(n_estimators=100, max_depth=15,
                                          min_samples_split=20, random_state=42,
                                          n_jobs=-1)
        model_name = "RandomForest"

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(y))
    for train_idx, val_idx in gkf.split(X, y, groups):
        estimator.fit(X[train_idx], y[train_idx])
        oof_pred[val_idx] = estimator.predict(X[val_idx])

    return y, oof_pred, model_name


def save(fig, name, tight=True):
    path = FIG_DIR / name
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# F1 — Comparación RMSE_CV: 4 modelos × 3 targets
# ═══════════════════════════════════════════════════════════════════════════════

def fig_model_comparison():
    print("\n[F1] Comparación de modelos (RMSE_CV)...")

    data = {
        "CR2_median": {"LinearRegression": 0.1274, "Ridge": 0.1274,
                       "RandomForest": 0.0681, "XGBoost": 0.0652},
        "CR2_p10":    {"LinearRegression": 0.1267, "Ridge": 0.1268,
                       "RandomForest": 0.0785, "XGBoost": 0.0755},
        "CR2_p90":    {"LinearRegression": 0.1838, "Ridge": 0.1838,
                       "RandomForest": 0.0981, "XGBoost": 0.0984},
    }

    # Intentar cargar desde CSV si existe
    csv_path = OUT_DIR / "results" / "rmse_cv_resumen.csv"
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path, index_col=0)
        for target in TARGETS:
            if target in df_csv.columns:
                for model in MODEL_ORDER:
                    if model in df_csv.index:
                        data[target][model] = df_csv.loc[model, target]

    n_models = len(MODEL_ORDER)
    n_targets = len(TARGETS)
    x = np.arange(n_targets)
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
        vals = [data[t].get(model, np.nan) for t in TARGETS]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=model,
                      color=color, edgecolor="white", linewidth=0.8, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([TARGETS[t] for t in TARGETS], fontsize=10)
    ax.set_ylabel("RMSE$_{CV}$ (GroupKFold, 5-fold)")
    ax.legend(ncol=4, loc="upper left")
    ax.set_ylim(0, max(data["CR2_p90"].values()) * 1.25)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Anotación de la diferencia entre lineal y XGBoost
    pct = (data["CR2_median"]["LinearRegression"] - data["CR2_median"]["XGBoost"]) \
          / data["CR2_median"]["LinearRegression"] * 100
    ax.annotate(f"↓{pct:.0f}% vs linear\nbaseline",
                xy=(0 + 1.5 * width, data["CR2_median"]["XGBoost"] + 0.003),
                xytext=(0.6, 0.11), textcoords="data",
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

    save(fig, "F1_model_comparison_rmse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F2 — Predicción vs valor real (OOF)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_predicted_vs_actual(agg, feature_cols):
    print("\n[F2] Predicted vs actual (out-of-fold)...")

    result = get_oof_predictions(agg, feature_cols, "CR2_median")
    if result[0] is None:
        print("  [SKIP] No se pudo computar OOF.")
        return

    y_true, y_pred, model_name = result
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Color por n_pisos
    n_pisos = agg["n_pisos_first"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Scatter predicted vs actual
    ax = axes[0]
    sc = ax.scatter(y_true, y_pred, c=n_pisos, cmap="RdYlGn_r",
                    alpha=0.3, s=6, rasterized=True)
    lim = [min(y_true.min(), y_pred.min()) - 0.05,
           max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed CR2$_{median}$")
    ax.set_ylabel("Predicted CR2$_{median}$ (OOF)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("n$_{pisos}$")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.97,
            f"R² = {r2:.3f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            fontsize=9)

    # Residual plot
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, c=n_pisos, cmap="RdYlGn_r",
                alpha=0.3, s=6, rasterized=True)
    ax2.axhline(0, color="k", linestyle="--", lw=1.2)
    ax2.axhline(rmse, color="#D6604D", linestyle=":", lw=1, label=f"+RMSE = {rmse:.4f}")
    ax2.axhline(-rmse, color="#D6604D", linestyle=":", lw=1, label=f"−RMSE = {rmse:.4f}")
    ax2.set_xlabel("Predicted CR2$_{median}$ (OOF)")
    ax2.set_ylabel("Residual (predicted − observed)")
    ax2.set_title("Residual analysis")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save(fig, "F2_predicted_vs_actual_oof.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F3 — Distribución de residuos
# ═══════════════════════════════════════════════════════════════════════════════

def fig_residual_distribution(agg, feature_cols):
    print("\n[F3] Distribución de residuos...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (target_col, label) in zip(axes, TARGETS.items()):
        result = get_oof_predictions(agg, feature_cols, target_col)
        if result[0] is None:
            ax.set_visible(False)
            continue
        y_true, y_pred, model_name = result
        residuals = y_pred - y_true
        rmse = np.sqrt(np.mean(residuals ** 2))

        ax.hist(residuals, bins=60, color=TARGET_COLORS[target_col],
                edgecolor="white", alpha=0.85, density=True)
        ax.axvline(0, color="k", lw=1.5, linestyle="--")
        ax.axvline(residuals.mean(), color="gray", lw=1, linestyle=":",
                   label=f"Mean = {residuals.mean():.4f}")

        # Curva normal de referencia
        from scipy.stats import norm
        xs = np.linspace(residuals.min(), residuals.max(), 300)
        ax.plot(xs, norm.pdf(xs, residuals.mean(), residuals.std()),
                "k-", lw=1.2, alpha=0.6, label="Normal fit")

        ax.set_xlabel("Residual (predicted − observed)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}\nRMSE = {rmse:.4f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "F3_residual_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F4 — Importancia de features: 3 targets en paralelo
# ═══════════════════════════════════════════════════════════════════════════════

def fig_feature_importance(agg, feature_cols):
    print("\n[F4] Feature importance (3 targets en paralelo)...")

    TOP_N = 12
    rename_map = {
        "N_Dampers_first": r"$N_d$",
        "Mf_KN_first": r"$M_f$ [kN·m]",
        "n_pisos_first": r"$n_p$",
        "T1_real_first": r"$T_1$ [s]",
        "densidad_dampers": "Density (N/n)",
        "posicion_promedio_dampers": "Mean position",
        "posicion_normalizada": "Norm. position",
        "concentracion_dampers": "Spread σ",
        "energia_total_normalizada": "Norm. energy",
    }
    for i in range(1, 21):
        rename_map[f"damper_piso{i}"] = f"Floor {i}"

    importances_per_target = {}
    for target_col in TARGETS:
        suffix = target_col.replace("CR2_", "")
        model = load_model("XGBoost", suffix) or load_model("RandomForest", suffix)
        if model is None:
            continue
        # Reentrenar para obtener importancias definitivas
        X = agg[feature_cols].values
        y = agg[target_col].values
        model.fit(X, y)
        importances_per_target[target_col] = model.feature_importances_

    if not importances_per_target:
        print("  [SKIP] No se pudieron cargar modelos.")
        return

    fig, axes = plt.subplots(1, len(importances_per_target),
                             figsize=(5 * len(importances_per_target), 6),
                             sharey=False)
    if len(importances_per_target) == 1:
        axes = [axes]

    for ax, (target_col, label) in zip(axes, TARGETS.items()):
        if target_col not in importances_per_target:
            continue
        # Top-N independiente para cada target
        imp_full = importances_per_target[target_col]
        top_idx = np.argsort(imp_full)[::-1][:TOP_N]
        feat_names = [rename_map.get(feature_cols[i], feature_cols[i]) for i in top_idx]
        imp = imp_full[top_idx]

        bars = ax.barh(range(TOP_N), imp, color=TARGET_COLORS[target_col],
                       alpha=0.85, edgecolor="white")
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels(feat_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Feature importance")
        ax.set_title(label)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, imp):
            ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=7.5)

    plt.tight_layout()
    save(fig, "F4_feature_importance_3targets.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F5 — CR2 vs densidad de amortiguadores (rendimiento decreciente)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_diminishing_returns(agg):
    print("\n[F5] Rendimiento decreciente vs densidad de amortiguadores...")

    # Para cada (estructura, N_Dampers), tomar la MEJOR config (min CR2_median)
    best = agg.groupby(["estructura_id", "N_Dampers_first", "n_pisos_first"])[
        "CR2_median"].min().reset_index()
    best["densidad"] = best["N_Dampers_first"] / best["n_pisos_first"]

    # Asignar grupo de altura
    def height_group(n):
        for label, (lo, hi) in HEIGHT_GROUPS.items():
            if lo <= n <= hi:
                return label
        return "Other"

    best["group"] = best["n_pisos_first"].apply(height_group)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))

    density_bins = np.linspace(0, 1, 21)
    bin_centers = 0.5 * (density_bins[:-1] + density_bins[1:])

    for (group_label, _), color in zip(HEIGHT_GROUPS.items(), HEIGHT_COLORS):
        sub = best[best["group"] == group_label]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["d_bin"] = pd.cut(sub["densidad"], bins=density_bins, labels=bin_centers,
                               include_lowest=True)
        stats = sub.groupby("d_bin", observed=True)["CR2_median"].agg(
            med="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).reset_index()
        stats["d_bin"] = stats["d_bin"].astype(float)
        stats = stats.dropna()
        ax.plot(stats["d_bin"], stats["med"], "o-", color=color,
                label=group_label, lw=1.8, ms=4)
        ax.fill_between(stats["d_bin"], stats["q25"], stats["q75"],
                        color=color, alpha=0.15)

    ax.set_xlabel(r"Coverage density  $\eta_d = N_d\,/\,n_p$")
    ax.set_ylabel(r"Best achievable $CR_{2,\mathrm{med}}$")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 1.02)

    plt.tight_layout()
    save(fig, "F5_diminishing_returns.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F6 — Curvas within-structure: mejora normalizada vs fracción cubierta
# ═══════════════════════════════════════════════════════════════════════════════

def fig_within_structure_curves(agg):
    print("\n[F6] Curvas within-structure (mejora relativa vs densidad)...")

    best = agg.groupby(["estructura_id", "N_Dampers_first", "n_pisos_first"])[
        "CR2_median"].min().reset_index()
    best["densidad"] = best["N_Dampers_first"] / best["n_pisos_first"]

    def height_group(n):
        for label, (lo, hi) in HEIGHT_GROUPS.items():
            if lo <= n <= hi:
                return label
        return "Other"

    best["group"] = best["n_pisos_first"].apply(height_group)

    # Para cada estructura, normalizar CR2 respecto al mínimo (con todos los dampers disponibles)
    cr2_min = best.groupby("estructura_id")["CR2_median"].min().rename("cr2_floor")
    cr2_max = best.groupby("estructura_id")["CR2_median"].max().rename("cr2_ceiling")
    best = best.merge(cr2_min, on="estructura_id").merge(cr2_max, on="estructura_id")
    best["cr2_norm"] = (best["CR2_median"] - best["cr2_floor"]) / \
                       (best["cr2_ceiling"] - best["cr2_floor"] + 1e-9)
    best["cr2_norm"] = best["cr2_norm"].clip(0, 1)

    # Invertir: 1 = peor (N=1), 0 = mejor (todos los amortiguadores)
    # En realidad queremos mostrar qué fracción de la mejora máxima se alcanza
    # "fracción_mejora" = 1 - cr2_norm = 0 con N=1, 1 con N=n_pisos
    best["frac_mejora"] = 1 - best["cr2_norm"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    density_bins = np.linspace(0, 1, 21)
    bin_centers = 0.5 * (density_bins[:-1] + density_bins[1:])

    for ax, (group_label, _), color in zip(axes, HEIGHT_GROUPS.items(), HEIGHT_COLORS):
        sub = best[best["group"] == group_label].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        # Trazar una muestra de curvas individuales (semi-transparente)
        sample_ids = sub["estructura_id"].unique()
        rng = np.random.default_rng(42)
        sample_ids_plot = rng.choice(sample_ids, size=min(80, len(sample_ids)), replace=False)
        for eid in sample_ids_plot:
            s = sub[sub["estructura_id"] == eid].sort_values("densidad")
            ax.plot(s["densidad"], s["frac_mejora"], color=color, alpha=0.10, lw=0.7)

        # Curva mediana
        sub["d_bin"] = pd.cut(sub["densidad"], bins=density_bins, labels=bin_centers,
                               include_lowest=True)
        stats = sub.groupby("d_bin", observed=True)["frac_mejora"].agg(
            med="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).reset_index()
        stats["d_bin"] = stats["d_bin"].astype(float)
        stats = stats.dropna()

        ax.plot(stats["d_bin"], stats["med"], "o-", color=color,
                lw=2.5, ms=5, label="Median", zorder=5)
        ax.fill_between(stats["d_bin"], stats["q25"], stats["q75"],
                        color=color, alpha=0.25, label="IQR", zorder=4)

        # Umbral del 80% de la mejora
        idx_80 = (stats["med"] >= 0.80).idxmax() if (stats["med"] >= 0.80).any() else None
        if idx_80 is not None:
            d_80 = stats.loc[idx_80, "d_bin"]
            ax.axvline(d_80, color="k", ls="--", lw=1, alpha=0.6,
                       label=f"80% gain at {d_80:.0%}")

        ax.axhline(0.80, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("Damper density  (N / n$_{pisos}$)")
        ax.set_ylabel("Fraction of maximum improvement")
        ax.set_title(f"{group_label}  (n = {sub['estructura_id'].nunique()} structures)")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save(fig, "F6_within_structure_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F7 — Mapa de calor interacción N_Dampers × Mf → CR2
# ═══════════════════════════════════════════════════════════════════════════════

def fig_interaction_heatmap(agg):
    print("\n[F7] Mapa de calor interacción N_Dampers × Mf (estratificado por altura)...")

    Mf_bins   = [0, 100, 200, 300, 400, 500, 9999]
    Mf_labels = ["<100", "100–200", "200–300", "300–400", "400–500", ">500"]
    Nd_bins   = [0, 2, 4, 6, 8, 10, 99]
    Nd_labels = ["1–2", "3–4", "5–6", "7–8", "9–10", "11+"]

    grupos = [
        ("Baja (3–5 pisos)",   (3,  5)),
        ("Media (6–12 pisos)", (6, 12)),
        ("Alta (13–20 pisos)", (13, 20)),
    ]

    agg2 = agg.copy()
    agg2["Mf_bin"] = pd.cut(agg2["Mf_KN_first"],    bins=Mf_bins, labels=Mf_labels)
    agg2["N_bin"]  = pd.cut(agg2["N_Dampers_first"], bins=Nd_bins, labels=Nd_labels)

    # Calcular vmin/vmax global para escala de color consistente
    vmin = agg2["CR2_median"].quantile(0.05)
    vmax = agg2["CR2_median"].quantile(0.95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (titulo, (pmin, pmax)) in zip(axes, grupos):
        sub = agg2[(agg2["n_pisos_first"] >= pmin) & (agg2["n_pisos_first"] <= pmax)]
        pivot = (sub.groupby(["N_bin", "Mf_bin"], observed=True)["CR2_median"]
                   .median().unstack())
        pivot = pivot.reindex(columns=[l for l in Mf_labels if l in pivot.columns])

        sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", annot=True, fmt=".3f",
                    linewidths=0.5, linecolor="white",
                    vmin=vmin, vmax=vmax,
                    cbar_kws={"label": r"Median $CR_{2,med}$"})
        ax.set_xlabel(r"$M_f$ [kN·m]")
        ax.set_ylabel(r"$N_d$ (number of dampers)")
        ax.set_title(titulo)

    plt.tight_layout()
    save(fig, "F7_interaction_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F8 — Análisis de convergencia extendido (muestra amplia)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_convergence_extended(agg, feature_cols):
    print("\n[F8] Análisis de convergencia extendido...")

    # Para cada estructura, ¿cuántos amortiguadores recomienda cada target?
    best_per_target = {}
    for target_col in TARGETS:
        suffix = target_col.replace("CR2_", "")
        model = load_model("XGBoost", suffix) or load_model("RandomForest", suffix)
        if model is None:
            continue
        agg_c = agg.copy()
        agg_c["pred"] = model.predict(agg[feature_cols].values)
        best_idx = agg_c.groupby("estructura_id")["pred"].idxmin()
        best = agg_c.loc[best_idx][["estructura_id", "N_Dampers_first", "n_pisos_first",
                                     "Mf_KN_first", target_col]].copy()
        best.rename(columns={
            "N_Dampers_first": f"N_{suffix}",
            "Mf_KN_first": f"Mf_{suffix}",
            target_col: f"CR2_{suffix}",
        }, inplace=True)
        best_per_target[target_col] = best

    if len(best_per_target) < 2:
        print("  [SKIP] Modelos insuficientes.")
        return

    # Merge
    df_conv = best_per_target["CR2_median"]
    for tc in ["CR2_p10", "CR2_p90"]:
        if tc in best_per_target:
            df_conv = df_conv.merge(best_per_target[tc], on=["estructura_id", "n_pisos_first"])

    if "N_p10" in df_conv.columns and "N_p90" in df_conv.columns:
        df_conv["N_agree"] = (
            (df_conv["N_median"] == df_conv["N_p10"]) &
            (df_conv["N_median"] == df_conv["N_p90"])
        )
        conv_rate = df_conv["N_agree"].mean() * 100
    else:
        conv_rate = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel izquierdo: distribución de N recomendado por target
    ax = axes[0]
    cols_N = [c for c in df_conv.columns if c.startswith("N_") and c != "N_agree"]
    colors_n = [TARGET_COLORS["CR2_median"], TARGET_COLORS["CR2_p10"], TARGET_COLORS["CR2_p90"]]
    target_labels = ["Median", "P10", "P90"]
    max_nd = int(df_conv[cols_N].max().max()) if cols_N else 20
    bins = np.arange(0.5, max_nd + 1.5)
    for col, color, lbl in zip(cols_N, colors_n, target_labels):
        if col not in df_conv.columns:
            continue
        ax.hist(df_conv[col], bins=bins, alpha=0.55, color=color,
                label=lbl, edgecolor="white")
    ax.set_xlabel("Recommended N$_{dampers}$")
    ax.set_ylabel("Number of structures")
    ax.legend()

    # Panel derecho: acuerdo N_median vs N_p90 por n_pisos
    ax2 = axes[1]
    if "N_p90" in df_conv.columns:
        df_conv["delta_N"] = (df_conv["N_p90"] - df_conv["N_median"]).abs()
        agree_by_height = df_conv.groupby("n_pisos_first")["N_agree"].mean() * 100
        delta_by_height = df_conv.groupby("n_pisos_first")["delta_N"].median()
        ax2b = ax2.twinx()
        ax2.bar(agree_by_height.index, agree_by_height.values,
                color="#2166AC", alpha=0.6, label="Agreement rate (%)")
        ax2b.plot(delta_by_height.index, delta_by_height.values,
                  "o-", color="#D6604D", lw=1.5, ms=4, label="|N_p90 − N_median|")
        ax2.set_xlabel("Building height (n$_{pisos}$)")
        ax2.set_ylabel("Agreement rate (%)", color="#2166AC")
        ax2b.set_ylabel("|ΔN| = |N$_{P90}$ − N$_{median}$|", color="#D6604D")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax2.set_visible(False)

    plt.tight_layout()
    save(fig, "F8_convergence_extended.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F9 — Panel de distribución de CR2 (para paper — sección resultados)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cr2_overview(agg):
    print("\n[F9] Panel de distribución de CR2 por grupo de altura...")

    def height_group(n):
        for label, (lo, hi) in HEIGHT_GROUPS.items():
            if lo <= n <= hi:
                return label
        return "Other"

    agg2 = agg.copy()
    agg2["group"] = agg2["n_pisos_first"].apply(height_group)

    # Para cada estructura: mejor CR2 disponible (toda la muestra)
    best = agg2.loc[agg2.groupby("estructura_id")["CR2_median"].idxmin()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Box-violin por grupo
    ax = axes[0]
    group_order = list(HEIGHT_GROUPS.keys())
    group_data = [best[best["group"] == g]["CR2_median"].values for g in group_order]
    vp = ax.violinplot(group_data, positions=range(len(group_order)),
                       showmedians=True, showextrema=False)
    for body, color in zip(vp["bodies"], HEIGHT_COLORS):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, fontsize=9)
    ax.set_ylabel("Best achievable CR2$_{median}$")
    ax.set_title("Best CR2 per structure by building height\n"
                 "(optimal damper configuration selected)")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="CR2 = 1 (neutral)")
    ax.legend(fontsize=9)

    # Scatter T1 vs CR2 coloreado por densidad óptima
    ax2 = axes[1]
    best["dens_opt"] = best["N_Dampers_first"] / best["n_pisos_first"]
    sc = ax2.scatter(best["T1_real_first"], best["CR2_median"],
                     c=best["dens_opt"], cmap="RdYlGn",
                     alpha=0.7, s=25, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Optimal damper density (N/n$_{pisos}$)")
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.set_xlabel("Fundamental period T₁ [s]")
    ax2.set_ylabel("Best achievable CR2$_{median}$")
    ax2.set_title("T₁ vs optimal CR2\n(color = optimal damper density)")

    plt.tight_layout()
    save(fig, "F9_cr2_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GENERACIÓN DE FIGURAS PARA EL PAPER")
    print("=" * 70)

    agg, feature_cols = load_data()
    print(f"Dataset: {agg.shape[0]:,} configuraciones, {agg['estructura_id'].nunique()} estructuras")

    fig_model_comparison()
    fig_predicted_vs_actual(agg, feature_cols)
    fig_residual_distribution(agg, feature_cols)
    fig_feature_importance(agg, feature_cols)
    fig_diminishing_returns(agg)
    fig_within_structure_curves(agg)
    fig_interaction_heatmap(agg)
    fig_convergence_extended(agg, feature_cols)
    fig_cr2_overview(agg)

    print(f"\n{'=' * 70}")
    print(f"Figuras generadas en: {FIG_DIR.resolve()}")
    print(f"{'=' * 70}")
    lines = [
        "Figuras producidas:",
        "  F1  F1_model_comparison_rmse.png       - RMSE_CV: 4 modelos x 3 targets",
        "  F2  F2_predicted_vs_actual_oof.png     - Prediccion vs real (OOF, XGBoost)",
        "  F3  F3_residual_distributions.png      - Distribucion de residuos",
        "  F4  F4_feature_importance_3targets.png - Feature importance en paralelo",
        "  F5  F5_diminishing_returns.png         - Rendimiento decreciente vs densidad",
        "  F6  F6_within_structure_curves.png     - Curvas within-structure por altura",
        "  F7  F7_interaction_heatmap.png         - Interaccion N x Mf -> CR2",
        "  F8  F8_convergence_extended.png        - Convergencia de targets (muestra completa)",
        "  F9  F9_cr2_overview.png               - Panel CR2 optimo por altura y T1",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
