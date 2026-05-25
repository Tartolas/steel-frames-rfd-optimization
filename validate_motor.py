"""
================================================================================
VALIDATE — Script corto para verificar que el motor funciona en TU PC
================================================================================
Antes de lanzar las 434k simulaciones de Fase 2 (que tomarán horas), corre este
script para verificar:

    1. Que tus 8 sismos cargan correctamente.
    2. Que el motor LCP-PSOR produce respuestas físicamente coherentes
       (R1, R2, R3 varían con Mf, CR2 < √3, etc.).
    3. Estimación honesta del tiempo total de cómputo.

Uso:
    python validate_motor.py
================================================================================
"""

import importlib.util
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Cargar el módulo Fase 2
spec = importlib.util.spec_from_file_location("phase2", "./02_run_simulations.py")
phase2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase2)


def main():
    print("=" * 70)
    print("VALIDACIÓN DEL MOTOR NSD-LCP-PSOR")
    print("=" * 70)

    # === 1) Verificar que existen los parquets de Fase 1 ===
    plan_dir = Path("./out_fase1")
    if not (plan_dir / "plan_estructuras_full.parquet").exists():
        print("[ERR] Falta plan_estructuras_full.parquet. Corre primero Fase 1:")
        print("      python 01_design_of_experiments.py")
        return

    # === 2) Verificar sismos ===
    print(f"\n[1/4] Cargando sismos desde ./sismos/...")
    try:
        sismos_data = phase2.cargar_todos_los_sismos()
        for sname, (dt, ag) in sismos_data.items():
            pga_g = float(np.max(np.abs(ag)) / 9.81)
            print(f"  [OK] {sname}: dt={dt:.4f}s, n={len(ag)}, PGA={pga_g:.3f} g")
    except Exception as e:
        print(f"  [ERR] {e}")
        print("\n  Asegúrate de que los 8 sismos están en ./sismos/ con uno de")
        print("  estos nombres: ElCentro 1940.csv, Jamiltepec 2018.csv, ...")
        return

    # === 3) Warmup Numba ===
    print(f"\n[2/4] Compilando funciones con Numba (puede tomar 30-60 s)...")
    t0 = time.time()
    phase2.warmup_numba(sismos_data)
    print(f"  Numba compilado en {time.time()-t0:.1f} s")

    # === 4) Probar barrido de Mf en una estructura ===
    print(f"\n[3/4] Validando física: barrido de Mf con todos los amortiguadores...")
    estr_full = pd.read_parquet("./out_fase1/plan_estructuras_full.parquet")
    # Tomar una estructura de 5 pisos
    estr = estr_full[estr_full["n_pisos"] == 5].iloc[0]
    import json
    m_vec = np.array(json.loads(estr["m_vec_kg"]))
    k_vec = np.array(json.loads(estr["k_vec_Nm"]))
    N = len(m_vec)
    M = np.diag(m_vec)
    K = phase2.build_K_shear(k_vec)
    C, w = phase2.rayleigh_C(M, K, 0.03, 1, 3)
    sname = list(sismos_data.keys())[0]
    dt, ag = sismos_data[sname]

    # Caso base
    u_b, v_b = phase2.nsd_core_rigid(
        dt, ag, C, K, m_vec,
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64))
    peak_u_b, peak_a_b, peak_Vb_b, drift_max_b, _ = phase2.calcular_metricas(
        u_b, v_b, dt, ag, m_vec)
    print(f"  Estructura {N} pisos, T1={estr['T1_real']:.2f}s, sismo: {sname}")
    print(f"  CASO BASE: peak_u_azotea={peak_u_b*100:.2f} cm")

    print(f"\n  {'Mf [KN]':>8} {'R1':>7} {'R2':>7} {'R3':>7} {'CR2':>7}")
    active = np.arange(N, dtype=np.int64)
    R1_list = []
    for Mf in [25, 100, 200, 500, 1000]:
        F_lim_val = Mf * 1000.0 / 0.40
        F_limits = np.full(len(active), F_lim_val)
        u, v = phase2.nsd_core_rigid(dt, ag, C, K, m_vec, active, F_limits)
        peak_u, peak_a, peak_Vb, drift_max, _ = phase2.calcular_metricas(
            u, v, dt, ag, m_vec)
        R1 = peak_u/peak_u_b; R2 = peak_a/peak_a_b; R3 = peak_Vb/peak_Vb_b
        CR2 = np.sqrt(R1**2 + R2**2 + R3**2)
        R1_list.append(R1)
        print(f"  {Mf:>8.0f} {R1:>7.4f} {R2:>7.4f} {R3:>7.4f} {CR2:>7.4f}")

    if max(R1_list) - min(R1_list) < 0.01:
        print(f"\n  [ERR] R1 no varía con Mf. Algo falla en el motor.")
        return
    else:
        print(f"\n  [OK] R1 varía claramente con Mf (rango {min(R1_list):.3f}-{max(R1_list):.3f})")

    # === 5) Benchmark de velocidad ===
    print(f"\n[4/4] Benchmark de velocidad (5 estructuras x 10 configs x 8 sismos)...")
    plan = pd.read_parquet("./out_fase1/plan_experimentos.parquet")
    # 5 estructuras de tamaños distintos
    sample_eids = []
    for npiso in [5, 10, 15]:
        sample_eids.extend(estr_full[estr_full["n_pisos"] == npiso]["estructura_id"].head(2).tolist())
    sample_eids = sample_eids[:5]

    n_sims = 0
    t0 = time.time()
    for eid in sample_eids:
        e = estr_full[estr_full["estructura_id"] == eid].iloc[0]
        plan_e = plan[plan["estructura_id"] == eid]
        cfg_ids = plan_e["config_local_id"].drop_duplicates().head(10).tolist()
        plan_e = plan_e[plan_e["config_local_id"].isin(cfg_ids)]
        df_res = phase2.simular_estructura(e, plan_e, sismos_data)
        n_sims += len(df_res)
    elapsed = time.time() - t0

    rate = n_sims / elapsed
    plan_total = pd.read_parquet("./out_fase1/plan_experimentos.parquet")
    n_total = len(plan_total)
    eta_h = n_total / rate / 3600
    print(f"  {n_sims} simulaciones en {elapsed:.1f} s -> {rate:.0f} sims/s")
    print(f"  Estimación para {n_total:,} simulaciones totales: {eta_h:.1f} h")
    print(f"  (Esta estimación NO incluye paralelismo entre estructuras.)")

    print("\n" + "=" * 70)
    print("[OK] VALIDACIÓN EXITOSA")
    print("=" * 70)
    print("\n  Ya puedes correr la Fase 2 completa con:")
    print("      python 02_run_simulations.py")
    print("\n  O en paralelo (si tienes muchos cores) con varias terminales:")
    print("      python 02_run_simulations.py --shard 0/4")
    print("      python 02_run_simulations.py --shard 1/4")
    print("      python 02_run_simulations.py --shard 2/4")
    print("      python 02_run_simulations.py --shard 3/4")


if __name__ == "__main__":
    main()
