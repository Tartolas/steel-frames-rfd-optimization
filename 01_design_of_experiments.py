"""
================================================================================
FASE 1 — DISEÑO DEL PLAN DE EXPERIMENTOS
================================================================================
Genera un plan completo de simulaciones para entrenar un modelo ML que prediga
el desempeño de marcos de acero acoplados a amortiguadores de fricción
rotacionales.

Estrategia:
    - Latin Hypercube Sampling (LHS) sobre el espacio de estructuras.
    - Cada estructura calibrada para alcanzar un T1_objetivo específico.
    - Configuraciones de amortiguadores por estructura: 40 LHS + 10+ estructuradas.
    - Parámetros sísmicos pre-calculados para los 8 registros.

Salidas:
    - plan_experimentos.parquet : tabla maestra con todas las simulaciones a correr
    - plan_estructuras.parquet  : una fila por estructura única (1000 filas)
    - plan_estructuras_full.parquet : incluye m_vec, k_vec serializados
    - plan_sismos.parquet       : parámetros de los 8 sismos
    - figs/                     : gráficos de validación de cobertura

Uso:
    python 01_design_of_experiments.py
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.stats import qmc
from pathlib import Path

# ========================================================================
# 0) CONFIGURACIÓN — ESPACIO DE DISEÑO
# ========================================================================

N_ESTRUCTURAS = 1000
N_PISOS_VALUES = [3, 5, 7, 10, 12, 15, 18, 20]
T1_RANGE = (0.3, 3.0)
MASA_RANGE = (150.0, 350.0)
PERFILES_MASA = ["constante", "azotea_reducida", "lineal_decreciente"]
PERFILES_RIGIDEZ = ["constante", "lineal_decreciente", "logaritmico", "bilineal"]

N_CONFIGS_LHS = 40
MF_RANGE = (25.0, 500.0)

SISMOS_DIR = "./sismos"
SISMOS_NAMES = [
    "ElCentro 1940", "Jamiltepec 2018", "Kobe 1995", "Loma Prieta 1989",
    "Niltepec 2017", "Northridge 1994", "Trinidad 1983", "Zacatula 1997"
]

RNG_SEED = 42
MASS_TON_TO_KG = 1000.0
G_MS2 = 9.81

OUT_DIR = Path("./out_fase1")
FIGS_DIR = OUT_DIR / "figs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIGS_DIR.mkdir(exist_ok=True, parents=True)


# ========================================================================
# 1) PERFILES Y MATRICES
# ========================================================================

def perfil_masa(N, tipo, masa_promedio):
    if tipo == "constante":
        m = np.ones(N)
    elif tipo == "azotea_reducida":
        m = np.ones(N)
        m[-1] = 0.7
    elif tipo == "lineal_decreciente":
        m = np.linspace(1.15, 0.85, N)
    else:
        raise ValueError(f"perfil_masa desconocido: {tipo}")
    m = m * (masa_promedio / m.mean())
    return m * MASS_TON_TO_KG


def perfil_rigidez(N, tipo):
    if tipo == "constante":
        k = np.ones(N)
    elif tipo == "lineal_decreciente":
        k = np.linspace(1.0, 0.4, N)
    elif tipo == "logaritmico":
        x = np.linspace(0, 1, N)
        k = np.exp(-1.5 * x)
    elif tipo == "bilineal":
        k = np.where(np.arange(N) < N // 2, 1.0, 0.5).astype(float)
    else:
        raise ValueError(f"perfil_rigidez desconocido: {tipo}")
    return k


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


def calcular_T1(M, K):
    vals, _ = sla.eigh(K, M)
    w1 = np.sqrt(max(vals[0], 1e-12))
    return 2 * np.pi / w1


def calibrar_k_para_T1(N, perfil_k_tipo, m_vec, T1_objetivo):
    """Calibra k_scale tal que T1_real = T1_objetivo. Solución analítica."""
    k_relativo = perfil_rigidez(N, perfil_k_tipo)
    K_PIVOTE = 1e8
    k_inicial = k_relativo * K_PIVOTE
    M = np.diag(m_vec)
    K = build_K_shear(k_inicial)
    T1_actual = calcular_T1(M, K)
    k_scale = (T1_actual / T1_objetivo) ** 2
    k_vec_final = k_inicial * k_scale
    K_final = build_K_shear(k_vec_final)
    T1_real = calcular_T1(M, K_final)
    return k_vec_final, T1_real


# ========================================================================
# 2) MUESTREO LHS DE ESTRUCTURAS
# ========================================================================

def muestreo_estructuras(n_samples, seed):
    sampler = qmc.LatinHypercube(d=5, seed=seed, scramble=True)
    U = sampler.random(n=n_samples)
    n_pisos = np.array([N_PISOS_VALUES[int(u * len(N_PISOS_VALUES))] for u in U[:, 0]])
    T1 = T1_RANGE[0] + U[:, 1] * (T1_RANGE[1] - T1_RANGE[0])
    masa = MASA_RANGE[0] + U[:, 2] * (MASA_RANGE[1] - MASA_RANGE[0])
    perfil_m = np.array([PERFILES_MASA[int(u * len(PERFILES_MASA))] for u in U[:, 3]])
    perfil_k = np.array([PERFILES_RIGIDEZ[int(u * len(PERFILES_RIGIDEZ))] for u in U[:, 4]])
    return pd.DataFrame({
        "estructura_id": np.arange(n_samples),
        "n_pisos": n_pisos,
        "T1_objetivo": T1,
        "masa_promedio_ton": masa,
        "perfil_masa": perfil_m,
        "perfil_rigidez": perfil_k,
    })


def expandir_estructuras_con_propiedades(df_estructuras):
    rows = []
    for _, row in df_estructuras.iterrows():
        N = int(row["n_pisos"])
        m_vec = perfil_masa(N, row["perfil_masa"], row["masa_promedio_ton"])
        k_vec, T1_real = calibrar_k_para_T1(N, row["perfil_rigidez"], m_vec, row["T1_objetivo"])
        rows.append({
            **row.to_dict(),
            "T1_real": T1_real,
            "T1_error_pct": 100.0 * abs(T1_real - row["T1_objetivo"]) / row["T1_objetivo"],
            "masa_total_ton": float(m_vec.sum() / MASS_TON_TO_KG),
            "k_promedio_Nm": float(k_vec.mean()),
            "m_vec_kg": json.dumps(m_vec.tolist()),
            "k_vec_Nm": json.dumps(k_vec.tolist()),
        })
    return pd.DataFrame(rows)


# ========================================================================
# 3) CONFIGURACIONES DE AMORTIGUADORES
# ========================================================================

def configs_estructuradas(N):
    configs = []
    for i in range(N):
        bits = np.zeros(N, dtype=int)
        bits[i] = 1
        configs.append(bits)
    configs.append(np.ones(N, dtype=int))
    bits = np.zeros(N, dtype=int); bits[: N // 2] = 1; configs.append(bits)
    bits = np.zeros(N, dtype=int); bits[N // 2:] = 1; configs.append(bits)
    return configs


def configs_lhs(N, n_samples, seed):
    sampler = qmc.LatinHypercube(d=2, seed=seed, scramble=True)
    U = sampler.random(n=n_samples)
    densidades = 0.1 + U[:, 0] * 0.9
    mfs = MF_RANGE[0] + U[:, 1] * (MF_RANGE[1] - MF_RANGE[0])
    rng = np.random.default_rng(seed)
    configs, mf_list = [], []
    for d, mf in zip(densidades, mfs):
        n_active = max(1, int(round(d * N)))
        idx = rng.choice(N, size=n_active, replace=False)
        bits = np.zeros(N, dtype=int); bits[idx] = 1
        configs.append(bits); mf_list.append(mf)
    return configs, mf_list


def muestreo_configs_por_estructura(N, seed_offset):
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    estr_configs = configs_estructuradas(N)
    estr_mfs = rng.uniform(MF_RANGE[0], MF_RANGE[1], size=len(estr_configs))
    lhs_configs, lhs_mfs = configs_lhs(N, N_CONFIGS_LHS, seed=RNG_SEED + seed_offset + 10000)
    all_configs = list(estr_configs) + list(lhs_configs)
    all_mfs = list(estr_mfs) + list(lhs_mfs)
    seen = set(); unique = []
    for bits, mf in zip(all_configs, all_mfs):
        key = ("".join(map(str, bits)), round(mf, 1))
        if key not in seen:
            seen.add(key); unique.append((bits, mf))
    return unique


# ========================================================================
# 4) PARÁMETROS SÍSMICOS
# ========================================================================

def calcular_parametros_sismicos(t, ag_g):
    ag = ag_g * G_MS2
    dt = float(np.median(np.diff(t)))
    n_steps = len(ag)
    PGA_g = float(np.max(np.abs(ag_g)))
    vg = np.cumsum(ag) * dt; vg = vg - np.mean(vg)
    PGV_ms = float(np.max(np.abs(vg)))
    PGV_PGA = float(PGV_ms / max(PGA_g * G_MS2, 1e-12))
    IA = float((np.pi / (2 * G_MS2)) * np.sum(ag ** 2) * dt)
    energia_acum = np.cumsum(ag ** 2) * dt
    energia_total = energia_acum[-1] if energia_acum[-1] > 0 else 1e-12
    norm = energia_acum / energia_total
    i5 = int(np.searchsorted(norm, 0.05))
    i95 = int(np.searchsorted(norm, 0.95))
    D5_95 = float((i95 - i5) * dt)
    fft = np.fft.rfft(ag)
    freqs = np.fft.rfftfreq(n_steps, d=dt)
    mask = (freqs >= 0.25) & (freqs <= 20.0)
    C = np.abs(fft[mask]); f_band = freqs[mask]
    if np.sum(C ** 2) > 0 and np.all(f_band > 0):
        Tm = float(np.sum((C ** 2) / f_band) / np.sum(C ** 2))
    else:
        Tm = float("nan")
    return {"PGA_g": PGA_g, "PGV_ms": PGV_ms, "PGV_PGA_s": PGV_PGA,
            "D5_95_s": D5_95, "Tm_s": Tm, "IA_ms": IA,
            "dt_s": dt, "n_steps": n_steps}


def cargar_y_caracterizar_sismos(sismos_dir, sismos_names):
    rows = []
    sismos_dir = Path(sismos_dir)
    for nombre in sismos_names:
        archivo = None
        for ext in [".csv", ".txt", ".dat", ".tsv"]:
            cand = sismos_dir / f"{nombre}{ext}"
            if cand.exists():
                archivo = cand; break
        if archivo is None:
            print(f"  [WARN] Sismo no encontrado: {nombre}")
            rows.append({"sismo": nombre, "archivo": None,
                         "PGA_g": np.nan, "PGV_ms": np.nan, "PGV_PGA_s": np.nan,
                         "D5_95_s": np.nan, "Tm_s": np.nan, "IA_ms": np.nan,
                         "dt_s": np.nan, "n_steps": -1})
            continue
        try:
            data = np.loadtxt(archivo, delimiter=",")
        except Exception:
            data = np.loadtxt(archivo)
        if data.ndim > 1 and data.shape[1] >= 2:
            t = data[:, 0]; ag_g = data[:, 1]
        else:
            ag_g = data; t = np.arange(len(ag_g)) * 0.02
        params = calcular_parametros_sismicos(t, ag_g)
        rows.append({"sismo": nombre, "archivo": str(archivo), **params})
    return pd.DataFrame(rows)


# ========================================================================
# 5) PLAN MAESTRO
# ========================================================================

def construir_plan_maestro(df_estructuras, df_sismos):
    filas = []; sim_id = 0
    print(f"\n  Generando configuraciones por estructura...")
    for _, est in df_estructuras.iterrows():
        N = int(est["n_pisos"])
        configs = muestreo_configs_por_estructura(N, seed_offset=int(est["estructura_id"]))
        for config_local_id, (bits, mf) in enumerate(configs):
            bits_str = "".join(map(str, bits))
            n_dampers = int(bits.sum())
            for _, sism in df_sismos.iterrows():
                filas.append({
                    "sim_id": sim_id, "estructura_id": int(est["estructura_id"]),
                    "config_local_id": config_local_id, "n_pisos": N,
                    "T1_objetivo": est["T1_objetivo"], "T1_real": est["T1_real"],
                    "masa_promedio_ton": est["masa_promedio_ton"],
                    "perfil_masa": est["perfil_masa"],
                    "perfil_rigidez": est["perfil_rigidez"],
                    "Bits": bits_str, "Mf_KN": mf, "N_Dampers": n_dampers,
                    "densidad_dampers": n_dampers / N, "sismo": sism["sismo"],
                })
                sim_id += 1
    return pd.DataFrame(filas)


# ========================================================================
# 6) GRÁFICOS DE VALIDACIÓN
# ========================================================================

def graficar_cobertura(df_estructuras, df_plan, figs_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df_estructuras["T1_objetivo"], bins=30, edgecolor="k")
    axes[0].set_xlabel("T1 objetivo [s]"); axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Cobertura de T1")
    axes[1].hist(df_estructuras["masa_promedio_ton"], bins=30, edgecolor="k")
    axes[1].set_xlabel("Masa promedio piso [ton]"); axes[1].set_title("Cobertura de masa")
    axes[2].hist(df_estructuras["n_pisos"], bins=np.arange(2.5, 21.5, 1), edgecolor="k")
    axes[2].set_xlabel("Número de pisos"); axes[2].set_title("Cobertura de n_pisos")
    plt.tight_layout(); plt.savefig(figs_dir / "01_cobertura_marginales.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(df_estructuras["T1_objetivo"], df_estructuras["n_pisos"],
                         c=df_estructuras["masa_promedio_ton"], cmap="viridis", alpha=0.6, s=15)
    axes[0].set_xlabel("T1 [s]"); axes[0].set_ylabel("n_pisos")
    axes[0].set_title("T1 vs n_pisos (color = masa)")
    plt.colorbar(sc, ax=axes[0], label="masa [ton]")
    axes[1].scatter(df_estructuras["T1_objetivo"], df_estructuras["masa_promedio_ton"],
                    c=df_estructuras["n_pisos"], cmap="plasma", alpha=0.6, s=15)
    axes[1].set_xlabel("T1 [s]"); axes[1].set_ylabel("masa [ton]")
    axes[1].set_title("T1 vs masa (color = n_pisos)")
    plt.tight_layout(); plt.savefig(figs_dir / "02_cobertura_conjunta.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_estructuras["perfil_masa"].value_counts().plot(kind="bar", ax=axes[0], edgecolor="k")
    axes[0].set_title("Distribución de perfil_masa"); axes[0].set_ylabel("Frecuencia")
    axes[0].tick_params(axis='x', rotation=20)
    df_estructuras["perfil_rigidez"].value_counts().plot(kind="bar", ax=axes[1], edgecolor="k")
    axes[1].set_title("Distribución de perfil_rigidez")
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout(); plt.savefig(figs_dir / "03_cobertura_categoricos.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_estructuras["T1_objetivo"], df_estructuras["T1_real"], alpha=0.5, s=15)
    lims = [df_estructuras["T1_objetivo"].min(), df_estructuras["T1_objetivo"].max()]
    ax.plot(lims, lims, "r--", lw=1, label="ideal")
    ax.set_xlabel("T1 objetivo [s]"); ax.set_ylabel("T1 real [s]")
    ax.set_title(f"Calibración T1 — error medio: {df_estructuras['T1_error_pct'].mean():.3f}%")
    ax.legend(); plt.tight_layout()
    plt.savefig(figs_dir / "04_calibracion_T1.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_plan["N_Dampers"], bins=30, edgecolor="k")
    axes[0].set_xlabel("Número de amortiguadores"); axes[0].set_title("Distribución N_Dampers")
    axes[1].hist(df_plan["Mf_KN"], bins=30, edgecolor="k")
    axes[1].set_xlabel("Mf [KN]"); axes[1].set_title("Distribución Mf")
    plt.tight_layout(); plt.savefig(figs_dir / "05_cobertura_amortiguadores.png", dpi=120); plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    N_demo = 10
    for ax, tipo in zip(axes[0], PERFILES_RIGIDEZ):
        k = perfil_rigidez(N_demo, tipo)
        ax.plot(k, np.arange(1, N_demo + 1), "-o")
        ax.set_title(f"Rigidez: {tipo}"); ax.set_xlabel("k_relativo")
        ax.set_ylabel("piso"); ax.grid(alpha=0.3)
    for ax, tipo in zip(axes[1], PERFILES_MASA + ["(vacío)"]):
        if tipo == "(vacío)": ax.axis("off"); continue
        m = perfil_masa(N_demo, tipo, masa_promedio=200) / MASS_TON_TO_KG
        ax.plot(m, np.arange(1, N_demo + 1), "-o", color="darkorange")
        ax.set_title(f"Masa: {tipo}"); ax.set_xlabel("masa [ton]")
        ax.set_ylabel("piso"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(figs_dir / "06_ejemplo_perfiles.png", dpi=120); plt.close()
    print(f"  [OK] Gráficos guardados en: {figs_dir.resolve()}")


# ========================================================================
# 7) MAIN
# ========================================================================

def main():
    print("=" * 70)
    print("FASE 1 — DISEÑO DEL PLAN DE EXPERIMENTOS")
    print("=" * 70)

    print(f"\n[1/5] Muestreando {N_ESTRUCTURAS} estructuras con LHS...")
    df_lhs = muestreo_estructuras(N_ESTRUCTURAS, seed=RNG_SEED)

    print(f"\n[2/5] Calibrando k_vec para T1_objetivo...")
    df_estructuras = expandir_estructuras_con_propiedades(df_lhs)
    err = df_estructuras["T1_error_pct"]
    print(f"      Error calibración T1: media={err.mean():.4f}%  max={err.max():.4f}%")

    print(f"\n[3/5] Caracterizando sismos...")
    df_sismos = cargar_y_caracterizar_sismos(SISMOS_DIR, SISMOS_NAMES)
    sismos_ok = df_sismos["archivo"].notna().sum()
    print(f"      Sismos cargados: {sismos_ok}/{len(SISMOS_NAMES)}")

    print(f"\n[4/5] Construyendo plan maestro...")
    df_plan = construir_plan_maestro(df_estructuras, df_sismos)
    print(f"      Filas en plan: {len(df_plan):,}")

    print(f"\n[5/5] Guardando salidas...")
    df_estructuras.drop(columns=["m_vec_kg", "k_vec_Nm"]).to_parquet(
        OUT_DIR / "plan_estructuras.parquet", index=False)
    df_estructuras.to_parquet(OUT_DIR / "plan_estructuras_full.parquet", index=False)
    df_sismos.to_parquet(OUT_DIR / "plan_sismos.parquet", index=False)
    df_plan.to_parquet(OUT_DIR / "plan_experimentos.parquet", index=False)
    graficar_cobertura(df_estructuras, df_plan, FIGS_DIR)

    print("\n" + "=" * 70)
    print("RESUMEN DEL PLAN GENERADO")
    print("=" * 70)
    print(f"  Estructuras únicas:        {len(df_estructuras):,}")
    print(f"  Filas (simulaciones):      {len(df_plan):,}")
    print(f"  Sismos por simulación:     {len(SISMOS_NAMES)}")
    print(f"\n  Archivos en {OUT_DIR.resolve()}:")
    print(f"    plan_estructuras.parquet")
    print(f"    plan_estructuras_full.parquet")
    print(f"    plan_sismos.parquet")
    print(f"    plan_experimentos.parquet")
    print(f"    figs/ (6 PNGs)")
    print("\n  [OK] Fase 1 completada.")


if __name__ == "__main__":
    main()
