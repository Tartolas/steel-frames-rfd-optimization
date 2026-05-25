# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> See also `../CLAUDE.md` for project-level context (user profile, seismic records, FEM modules).

---

## Pipeline scripts (Proyecto/)

| Script | Phase | Runtime |
|--------|-------|---------|
| `01_design_of_experiments.py` | 1 — DoE | ~30 s |
| `02_run_simulations.py` | 2 — Simulation | 1–3 h |
| `validate_motor.py` | 3a — Validate | ~60 s |
| `inspect_plan.py` | 3b — Inspect | ~10 s |
| `04_full_pipeline.py` | 4 — ML | ~5 min |
| `05_paper_figures.py` | 5 — Figures | ~2 min |
| `generate_paper.py` | 6 — LaTeX | ~1 min |
| `optimizador_pareto.py` | Opt — Pareto | stub |
| `demo_app.py` | Viz — Interactive | server |

---

## Commands

```bash
# Phase 1
python 01_design_of_experiments.py

# Phase 2 (full, ~434k sims)
python 02_run_simulations.py

# Phase 2 parallelized (4 shards)
python 02_run_simulations.py --shard 0 4
python 02_run_simulations.py --shard 1 4
python 02_run_simulations.py --shard 2 4
python 02_run_simulations.py --shard 3 4

# Phase 2 consolidate shards → all_results.parquet
python 02_run_simulations.py --consolidate

# Phase 3 — validate before long run
python validate_motor.py
python inspect_plan.py

# Phase 4
python 04_full_pipeline.py

# Figures and paper
python 05_paper_figures.py
python generate_paper.py
```

---

## I/O per phase

```
Sismos/ ──────────────────────────────────────┐
                                              ↓
01_design_of_experiments.py → out_fase1/
    plan_estructuras.parquet        # 1000 rows; no serialized vectors
    plan_estructuras_full.parquet   # + m_vec_kg/k_vec_Nm as JSON columns
    plan_sismos.parquet             # 8 rows with PGA, PGV, D5_95, Tm, IA
    plan_experimentos.parquet       # ~434 992 rows: one per (struct, cfg, sismo)
    figs/                           # 6 PNGs: LHS coverage, T1 calibration

02_run_simulations.py ← out_fase1/ + Sismos/
    → out_fase2/
        struct_NNNNN.parquet        # one per structure; ~430–500 rows
        all_results.parquet         # consolidated; ~434 k rows, 25+ columns
        done_NNNNN.flag             # checkpoint; delete to rerun that struct

04_full_pipeline.py ← out_fase2/all_results.parquet
    → out_ml/
        dataset_ml.parquet          # aggregated + engineered features
        eda/                        # distribution/correlation plots
        models/                     # *.pkl per (algorithm, target)
        results/
            model_comparison_*.csv
            feature_importance_*.png
            convergencia_targets.csv
            convergencia_targets.png
```

---

## Core simulation algorithm — NSD-LCP-PSOR (`02_run_simulations.py`)

`nsd_core_rigid()` is Numba-compiled (`@njit`). Do **not** add Python objects or dynamic types inside it.

**Integrator**: implicit Newmark-β, θ=0.5 (average acceleration, unconditionally stable).

**Friction LCP**: at each timestep, find λ ∈ [−F_lim, +F_lim] solving:
```
W·λ + q ≥ 0,  (W·λ + q)·λ = 0
```
W = explicit inverse of condensed effective-mass at damper DOFs.

**PSOR update** (`max_iter=20`, `tol=1e-6`, `ω=1.0`):
```
λ_i ← clip( −(q_i + Σ_{j≠i} W_{ij}·λ_j) / W_{ii}, −F_lim_i, +F_lim_i )
```

**Key parameters** (hardcoded in script, change with care):
| Symbol | Value | Meaning |
|--------|-------|---------|
| `zeta` | 0.03 | Rayleigh damping ratio |
| Rayleigh modes | 1, 3 | Modes used to compute a₀, a₁ |
| `h_a` | 0.40 m | RFD lever arm |
| `max_iter` | 20 | PSOR iterations per timestep |
| `tol` | 1e-6 | PSOR convergence |

**Metrics computed per simulation** (columns in all_results.parquet):
```
R1 (disp ratio), R2 (accel ratio), R3 (base-shear ratio)
CR1 = √(R1²+R2²),  CR2 = √(R1²+R2²+R3²)
peak_u_azotea, peak_a_azotea, peak_Vb, drift_max, drift_ratio
E_total_J, E_promedio_damper_J
```

---

## Phase 1 design space

LHS over:
- `n_pisos` ∈ {3, 5, 7, 10, 12, 15, 18, 20}
- `T1_objetivo` ∈ [0.3, 3.0] s
- `masa_promedio` ∈ [150, 350] ton/piso
- `perfil_masa` ∈ {const, roof_red, linear_dec}
- `perfil_rigidez` ∈ {const, lin_dec, log, bilin}

Per structure: ~50 RFD configs = 10 structured (single-floor + all + halves) + 40 LHS (placement bits + Mf ∈ [25, 500] kN).

T1 calibration: `calibrar_k_para_T1()` scales k_vec analytically until eigenvalue matches T1_objetivo within tolerance; T1_error_pct is logged — values >5% indicate poorly-conditioned cases.

---

## Phase 4 ML architecture

**Aggregation**: group by `(estructura_id, config_local_id)` over 8 earthquakes → CR2 median, p10, p90 (+ mean, std, min, max); also aggregate R1, R2, R3, drift_ratio, E_total_J (median).

**Targets**: `CR2_median`, `CR2_p10`, `CR2_p90`.

**Feature engineering** (~20–25 features):
```python
# Per-floor binary indicators
damper_pisoX  for X in 1..n_pisos   # one-hot placement

# Derived
densidad_dampers = N_Dampers / n_pisos
Mf_por_damper = Mf_KN / max(N_Dampers, 1)
posicion_promedio_dampers = mean(active floor indices) / n_pisos
concentracion_dampers = std(active floor indices) / n_pisos
energia_total_normalizada = E_total_J / (n_pisos × 1e6)
```

**CV**: `GroupKFold(n_splits=5)` grouped by `estructura_id` — mandatory to prevent leakage (same structure appears across multiple configs).

**Models**:
- `LinearRegression`, `Ridge(alpha=1.0)` — baselines
- `RandomForestRegressor(n_estimators=100, max_depth=15)`
- `HistGradientBoostingRegressor` or XGBoost if available

**Convergence analysis**: for 20 random structures, optimal config under each target is predicted; convergence rate = fraction with identical design across CR2_median, CR2_p10, CR2_p90.

---

## Shear-frame K assembly

```python
# build_K_shear(k_vec)  — k_vec length = n_pisos
K[i, i]   = k_vec[i] + (k_vec[i+1] if i < n-1 else 0)
K[i, i±1] = -k_vec[i±1]
# tridiagonal, units N/m
```

Rayleigh C:
```
a0 = 2ζ·ω₁·ω₃ / (ω₁+ω₃),  a1 = 2ζ / (ω₁+ω₃)
C = a0·M + a1·K
```

---

## Numba considerations

`nsd_core_rigid()` uses `@njit`. Rules:
- All arrays must be pre-allocated NumPy arrays passed as arguments.
- No Python lists, dicts, or objects inside the function.
- First call triggers JIT compilation (~5–10 s warmup); `validate_motor.py` does this explicitly.
- After editing `nsd_core_rigid`, delete Numba cache (`__pycache__`) if behavior seems stale.

---

## Checkpointing (Phase 2)

Phase 2 writes `out_fase2/done_NNNNN.flag` after each completed structure. On restart, structures with flags are skipped. To rerun specific structures, delete the corresponding flag files. To consolidate after sharded runs: `--consolidate` merges all `struct_*.parquet` into `all_results.parquet`.

---

## Seismic records

Files in `Sismos/`: two-column `.txt` (time [s], acceleration [g or m/s²]). `cargar_sismo()` auto-detects units and resamples to uniform dt. Records: ElCentro 1940, Kobe 1995, Northridge 1994, Loma Prieta 1989, Zacatula 1997, Jamiltepec 2018, Niltepec 2017, Trinidad 1983.
