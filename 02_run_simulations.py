"""
================================================================================
FASE 2 — MOTOR DE SIMULACIÓN MASIVA (motor de tesis: NSD-LCP-PSOR)
================================================================================
Lee el plan de Fase 1 y ejecuta TODAS las simulaciones usando el motor que
fue validado en la tesis original: `nsd_core_rigid` con LCP-PSOR.

DIFERENCIAS RESPECTO AL MOTOR Namorado-SDOF (Fig 6/7):
  - El motor de tesis usa LCP (Linear Complementarity Problem) en vez de
    iteración punto fijo simple, lo que SÍ funciona para MDOF.
  - Variables: λ (impulsos por amortiguador) en vez de i_f por piso.
  - Matriz W = H^T M_hat^-1 H * dt explícita, bien condicionada.
  - Sin parámetro α: PSOR resuelve directamente.
  - Acepta active_indices = [] (caso base trivial).

PARÁMETROS DEL EXPERIMENTO (decididos):
  - Rayleigh damping en modos 1 y 3, ζ = 0.03
  - h_a = 0.40 m constante (brazo de palanca para Mf -> Ff)
  - PSOR: max_iter=20, tol=1e-6 (versión rápida)
  - Métricas: R1, R2, R3, CR1, CR2 + drift_max + drift_ratio + peak_u_global
              + E_total estimada en post-proceso

CHECKPOINTING: 1 parquet por estructura.

Uso:
    python 02_run_simulations.py
    python 02_run_simulations.py --shard 0/4
    python 02_run_simulations.py --consolidate
================================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

try:
    from numba import njit
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k): return x


# ========================================================================
# 0) CONFIGURACIÓN
# ========================================================================

PLAN_DIR = Path("./out_fase1")
RESULTS_DIR = Path("./out_fase2")
SISMOS_DIR = Path("./sismos")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DAMPING_RATIO = 0.03
RAYLEIGH_MODE_I = 1
RAYLEIGH_MODE_J = 3
H_A_M = 0.40
G_MS2 = 9.81

PSOR_MAX_ITER = 20
PSOR_TOL = 1e-6
EPS_DEN = 1e-12

SISMOS_NAMES = [
    "ElCentro 1940", "Jamiltepec 2018", "Kobe 1995", "Loma Prieta 1989",
    "Niltepec 2017", "Northridge 1994", "Trinidad 1983", "Zacatula 1997"
]


# ========================================================================
# 1) MOTOR NSD — LCP-PSOR (DE LA TESIS, COPIADO TAL CUAL)
# ========================================================================

@njit(fastmath=True, cache=False)
def solve_lcp_psor(W, q, limits, x_init, max_iter=20, tol=1e-6):
    n = len(q)
    x = np.copy(x_init)
    omega = 1.0
    for _ in range(max_iter):
        x_old = np.copy(x)
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if i != j:
                    sigma += W[i, j] * x[j]
            val = -(q[i] + sigma) / W[i, i]
            val = (1.0 - omega) * x_old[i] + omega * val
            lim = limits[i]
            if val > lim:
                val = lim
            elif val < -lim:
                val = -lim
            x[i] = val
        diff = 0.0
        for i in range(n):
            d = abs(x[i] - x_old[i])
            if d > diff:
                diff = d
        if diff < tol:
            break
    return x


@njit(fastmath=True, cache=False, nogil=True)
def nsd_core_rigid(dt, ag, C, K_frame, m_vec, active_indices, F_limits):
    """Motor de tesis. Devuelve (u, v): desplazamientos y velocidades."""
    N = len(m_vec)
    n_steps = len(ag)
    theta = 0.5
    dt_th = dt * theta
    dt_th2 = dt_th * dt_th

    M_hat = np.zeros((N, N))
    for i in range(N):
        M_hat[i, i] = m_vec[i]
    for i in range(N):
        for j in range(N):
            M_hat[i, j] += dt_th * C[i, j] + dt_th2 * K_frame[i, j]
    M_hat_inv = np.linalg.inv(M_hat)

    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = m_vec[i]
    for i in range(N):
        for j in range(N):
            A[i, j] -= dt_th * C[i, j] + dt_th2 * K_frame[i, j]

    n_active = len(active_indices)
    W = np.zeros((n_active, n_active))
    for r in range(n_active):
        rf = active_indices[r]
        for c in range(n_active):
            cf = active_indices[c]
            term = M_hat_inv[rf, cf]
            if rf > 0:
                term -= M_hat_inv[rf-1, cf]
            if cf > 0:
                term -= M_hat_inv[rf, cf-1]
            if rf > 0 and cf > 0:
                term += M_hat_inv[rf-1, cf-1]
            W[r, c] = term * dt

    u = np.zeros((n_steps, N))
    v = np.zeros((n_steps, N))

    lambda_curr = np.zeros(n_active)
    u_curr = np.zeros(N)
    v_curr = np.zeros(N)

    RHS = np.zeros(N)
    Ku = np.zeros(N)
    v_free = np.zeros(N)
    q_vec = np.zeros(n_active)
    f_corr = np.zeros(N)
    Minv_fc = np.zeros(N)

    for k in range(n_steps - 1):
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += K_frame[i, j] * u_curr[j]
            Ku[i] = s

        ag_avg = ag[k+1] + ag[k]
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += A[i, j] * v_curr[j]
            RHS[i] = s - dt * Ku[i] - dt_th * m_vec[i] * ag_avg

        for i in range(N):
            s = 0.0
            for j in range(N):
                s += M_hat_inv[i, j] * RHS[j]
            v_free[i] = s

        for r in range(n_active):
            fl = active_indices[r]
            gr = v_free[fl]
            if fl > 0:
                gr -= v_free[fl-1]
            q_vec[r] = gr

        if n_active > 0:
            lambda_new = solve_lcp_psor(W, q_vec, F_limits, lambda_curr,
                                         max_iter=PSOR_MAX_ITER, tol=PSOR_TOL)
        else:
            lambda_new = lambda_curr

        for i in range(N):
            f_corr[i] = 0.0
        for r in range(n_active):
            fl = active_indices[r]
            val = lambda_new[r]
            f_corr[fl] += val
            if fl > 0:
                f_corr[fl-1] -= val

        for i in range(N):
            s = 0.0
            for j in range(N):
                s += M_hat_inv[i, j] * f_corr[j]
            Minv_fc[i] = s

        for i in range(N):
            v_next = v_free[i] + Minv_fc[i] * dt
            v[k+1, i] = v_next
            u[k+1, i] = u_curr[i] + dt * ((1.0 - theta) * v_curr[i] + theta * v_next)

        for i in range(N):
            u_curr[i] = u[k+1, i]
            v_curr[i] = v[k+1, i]
        for r in range(n_active):
            lambda_curr[r] = lambda_new[r]

    return u, v


# ========================================================================
# 2) UTILIDADES
# ========================================================================

def build_K_shear(k_vec):
    N = len(k_vec)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = k_vec[i] + (k_vec[i+1] if i < N-1 else 0.0)
        if i > 0:
            K[i, i-1] = -k_vec[i]
        if i < N-1:
            K[i, i+1] = -k_vec[i+1]
    return K


def rayleigh_C(M, K, zeta, mode_i, mode_j):
    vals, _ = sla.eigh(K, M)
    w = np.sqrt(np.maximum(vals, 0.0))
    i = mode_i - 1
    j = mode_j - 1
    if j >= len(w):
        w1 = float(w[0])
        a0 = 2.0 * zeta * w1
        a1 = 0.0
    else:
        wi = float(w[i])
        wj = float(w[j])
        a0 = 2.0 * zeta * wi * wj / (wi + wj)
        a1 = 2.0 * zeta / (wi + wj)
    return a0 * M + a1 * K, w


def cargar_sismo(path):
    try:
        d = np.loadtxt(path, delimiter=",")
    except Exception:
        d = np.loadtxt(path)
    if d.ndim > 1 and d.shape[1] >= 2:
        t = d[:, 0].astype(np.float64)
        ag = d[:, 1].astype(np.float64) * G_MS2
        dt = float(np.median(np.diff(t)))
    else:
        ag = d.astype(np.float64) * G_MS2
        dt = 0.02
    if (not np.isfinite(dt)) or dt <= 0:
        dt = 0.02
    if not np.all(np.isfinite(ag)):
        ag = ag[np.isfinite(ag)]
    return dt, ag


# ========================================================================
# 3) MÉTRICAS
# ========================================================================

def calcular_metricas(u, v, dt, ag, m_vec):
    peak_u = float(np.max(np.abs(u[:, -1])))
    a_rel = (v[1:] - v[:-1]) / dt
    a_abs = a_rel + ag[1:].reshape(-1, 1)
    peak_a = float(np.max(np.abs(a_abs[:, -1])))
    Vb_t = np.abs(a_abs @ m_vec)
    peak_Vb = float(np.max(Vb_t))
    if u.shape[1] >= 2:
        drifts = np.abs(np.diff(u, axis=1))
        drift_max = float(np.max(drifts))
    else:
        drift_max = float(np.max(np.abs(u)))
    peak_u_global = float(np.max(np.abs(u)))
    return peak_u, peak_a, peak_Vb, drift_max, peak_u_global


def energia_disipada_post(u, active_indices, F_limits):
    """
    Energía disipada por amortiguadores (estimación post-proceso).
    Cota superior: E_j ~ F_lim_j * sum_t |Δu_rel_j(t)|.
    Sirve como feature adicional para ML.
    """
    if len(active_indices) == 0:
        return 0.0
    E = 0.0
    for r in range(len(active_indices)):
        j = int(active_indices[r])
        F_lim = float(F_limits[r])
        if j == 0:
            u_rel = u[:, 0]
        else:
            u_rel = u[:, j] - u[:, j - 1]
        du_rel = np.abs(np.diff(u_rel))
        E += F_lim * float(np.sum(du_rel))
    return float(E)


# ========================================================================
# 4) SIMULAR UNA ESTRUCTURA
# ========================================================================

def simular_estructura(estr_row, plan_estr, sismos_data):
    N = int(estr_row["n_pisos"])
    m_vec = np.array(json.loads(estr_row["m_vec_kg"]), dtype=np.float64)
    k_vec = np.array(json.loads(estr_row["k_vec_Nm"]), dtype=np.float64)
    M = np.diag(m_vec)
    K = build_K_shear(k_vec)
    C, w = rayleigh_C(M, K, DAMPING_RATIO, RAYLEIGH_MODE_I, RAYLEIGH_MODE_J)
    T1_real = 2 * np.pi / max(w[0], EPS_DEN)

    # Caso base por sismo
    base_per_sismo = {}
    for sname, (dt, ag) in sismos_data.items():
        u_b, v_b = nsd_core_rigid(
            dt, ag, C, K, m_vec,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64))
        peak_u, peak_a, peak_Vb, drift_max, peak_u_glob = calcular_metricas(
            u_b, v_b, dt, ag, m_vec)
        base_per_sismo[sname] = {
            "dt": dt, "ag": ag,
            "peak_u_base": peak_u, "peak_a_base": peak_a, "peak_Vb_base": peak_Vb,
            "drift_max_base": drift_max, "peak_u_glob_base": peak_u_glob,
        }

    configs_df = plan_estr[["config_local_id", "Bits", "Mf_KN", "N_Dampers"]].drop_duplicates(
        subset=["config_local_id"])

    rows = []
    for _, c in configs_df.iterrows():
        bits_str = c["Bits"]
        Mf = float(c["Mf_KN"])
        n_d = int(c["N_Dampers"])

        bits = np.array([int(ch) for ch in bits_str], dtype=np.int64)
        active_indices = np.where(bits == 1)[0].astype(np.int64)
        F_lim_val = Mf * 1000.0 / H_A_M
        F_limits = np.full(len(active_indices), F_lim_val, dtype=np.float64)

        for sname, base in base_per_sismo.items():
            u, v = nsd_core_rigid(
                base["dt"], base["ag"], C, K, m_vec,
                active_indices, F_limits)
            peak_u, peak_a, peak_Vb, drift_max, peak_u_glob = calcular_metricas(
                u, v, base["dt"], base["ag"], m_vec)

            R1 = peak_u / base["peak_u_base"] if base["peak_u_base"] > 0 else 1.0
            R2 = peak_a / base["peak_a_base"] if base["peak_a_base"] > 0 else 1.0
            R3 = peak_Vb / base["peak_Vb_base"] if base["peak_Vb_base"] > 0 else 1.0
            CR1 = float(np.sqrt(R1**2 + R2**2))
            CR2 = float(np.sqrt(R1**2 + R2**2 + R3**2))

            E_total = energia_disipada_post(u, active_indices, F_limits)
            E_promedio_damper = E_total / n_d if n_d > 0 else 0.0
            drift_ratio = drift_max / base["drift_max_base"] if base["drift_max_base"] > 0 else 1.0

            rows.append({
                "estructura_id": int(estr_row["estructura_id"]),
                "config_local_id": int(c["config_local_id"]),
                "sismo": sname,
                "n_pisos": N,
                "T1_real": float(T1_real),
                "Bits": bits_str,
                "Mf_KN": Mf,
                "N_Dampers": n_d,
                "peak_u_azotea_m": peak_u,
                "peak_a_azotea_ms2": peak_a,
                "peak_Vb_N": peak_Vb,
                "drift_max_m": drift_max,
                "peak_u_global_m": peak_u_glob,
                "peak_u_base_m": base["peak_u_base"],
                "peak_a_base_ms2": base["peak_a_base"],
                "peak_Vb_base_N": base["peak_Vb_base"],
                "drift_max_base_m": base["drift_max_base"],
                "R1": R1, "R2": R2, "R3": R3,
                "CR1": CR1, "CR2": CR2,
                "drift_ratio": drift_ratio,
                "E_total_J": E_total,
                "E_promedio_damper_J": E_promedio_damper,
            })

    return pd.DataFrame(rows)


# ========================================================================
# 5) ORQUESTACIÓN
# ========================================================================

def cargar_todos_los_sismos():
    sismos = {}
    for nombre in SISMOS_NAMES:
        archivo = None
        for ext in [".csv", ".txt", ".dat", ".tsv"]:
            cand = SISMOS_DIR / f"{nombre}{ext}"
            if cand.exists():
                archivo = cand; break
        if archivo is None:
            raise FileNotFoundError(f"No se encontró el sismo: {nombre} en {SISMOS_DIR}")
        dt, ag = cargar_sismo(archivo)
        sismos[nombre] = (dt, ag)
    return sismos


def warmup_numba(sismos_data):
    if not NUMBA_OK:
        return
    print("  Warmup Numba (compilación inicial)...")
    N = 3
    m_vec = np.full(N, 200000.0)
    k_vec = np.full(N, 1e8)
    M = np.diag(m_vec); K = build_K_shear(k_vec)
    C, _ = rayleigh_C(M, K, DAMPING_RATIO, RAYLEIGH_MODE_I, RAYLEIGH_MODE_J)
    sname = next(iter(sismos_data))
    dt, ag = sismos_data[sname]
    active = np.array([0, 2], dtype=np.int64)
    F_lim = np.array([1e5, 1e5], dtype=np.float64)
    _ = nsd_core_rigid(dt, ag[:200], C, K, m_vec, active, F_lim)
    _ = nsd_core_rigid(dt, ag[:200], C, K, m_vec,
                       np.array([], dtype=np.int64),
                       np.array([], dtype=np.float64))
    print("  [OK] Numba listo.\n")


def run_phase2(shard_idx=0, shard_total=1):
    print("=" * 70)
    print("FASE 2 — SIMULACIÓN MASIVA (motor NSD-LCP-PSOR de tesis)")
    print("=" * 70)

    if not NUMBA_OK:
        print("[WARN] Numba NO disponible. pip install numba")

    print(f"\n  PLAN_DIR    : {PLAN_DIR.resolve()}")
    print(f"  RESULTS_DIR : {RESULTS_DIR.resolve()}")
    print(f"  SISMOS_DIR  : {SISMOS_DIR.resolve()}")
    print(f"  Shard       : {shard_idx + 1}/{shard_total}")
    print(f"  Solver      : PSOR max_iter={PSOR_MAX_ITER}, tol={PSOR_TOL}, "
          f"ζ={DAMPING_RATIO}, Rayleigh modos=({RAYLEIGH_MODE_I},{RAYLEIGH_MODE_J})")
    print(f"  h_a         : {H_A_M} m")

    estr_full = pd.read_parquet(PLAN_DIR / "plan_estructuras_full.parquet")
    plan = pd.read_parquet(PLAN_DIR / "plan_experimentos.parquet")
    print(f"\n  Estructuras: {len(estr_full)}  Filas:  {len(plan)}")

    print(f"\n  Cargando {len(SISMOS_NAMES)} sismos...")
    sismos_data = cargar_todos_los_sismos()
    print(f"  [OK] Sismos cargados.")
    warmup_numba(sismos_data)

    estr_ids = sorted(estr_full["estructura_id"].unique())
    estr_ids_shard = [e for e in estr_ids if e % shard_total == shard_idx]
    print(f"  Estructuras en shard: {len(estr_ids_shard)}")

    pendientes = [eid for eid in estr_ids_shard
                  if not (RESULTS_DIR / f"_DONE_struct_{eid:05d}.flag").exists()]
    print(f"  Pendientes:           {len(pendientes)}")
    if not pendientes:
        print("\n  [OK] Nada que hacer.")
        return

    t0 = time.time()
    n_filas_total = 0
    for k, eid in enumerate(tqdm(pendientes, desc="Estructuras", unit="estr")):
        estr_row = estr_full[estr_full["estructura_id"] == eid].iloc[0]
        plan_estr = plan[plan["estructura_id"] == eid]
        df_res = simular_estructura(estr_row, plan_estr, sismos_data)
        out_path = RESULTS_DIR / f"struct_{eid:05d}.parquet"
        df_res.to_parquet(out_path, index=False)
        (RESULTS_DIR / f"_DONE_struct_{eid:05d}.flag").touch()
        n_filas_total += len(df_res)
        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (len(pendientes) - k - 1) / rate if rate > 0 else float("inf")
            tqdm.write(f"  [{k+1}/{len(pendientes)}] eid={eid:5d} | "
                       f"{len(df_res)} filas | ritmo {rate*3600:.0f} estr/h | "
                       f"ETA {eta/3600:.1f} h")

    elapsed = time.time() - t0
    print(f"\n  [OK] Shard completado.")
    print(f"     Tiempo:        {elapsed/3600:.2f} h")
    print(f"     Filas escritas: {n_filas_total:,}")


def consolidate_results():
    print("\nConsolidando resultados...")
    files = sorted(RESULTS_DIR.glob("struct_*.parquet"))
    if not files:
        print("  [ERR] No hay archivos para consolidar.")
        return
    dfs = [pd.read_parquet(f) for f in tqdm(files, desc="Leyendo")]
    big = pd.concat(dfs, ignore_index=True)
    out = RESULTS_DIR / "all_results.parquet"
    big.to_parquet(out, index=False)
    print(f"  [OK] Consolidado: {out}")
    print(f"     Filas:  {len(big):,}")
    print(f"     Tamaño: {out.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, default="0/1")
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()

    if args.consolidate:
        consolidate_results()
        return
    idx_str, total_str = args.shard.split("/")
    run_phase2(shard_idx=int(idx_str), shard_total=int(total_str))


if __name__ == "__main__":
    main()
