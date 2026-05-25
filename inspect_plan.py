"""
================================================================================
INSPECT_PLAN — Inspector del plan de experimentos generado en Fase 1
================================================================================
Carga los parquets de Fase 1 y genera estadísticas, resúmenes y plots
exploratorios para validar el plan ANTES de lanzar las 434k simulaciones.

Ubicación esperada (rutas relativas al directorio donde corres el script):
    ./out_fase1/plan_estructuras.parquet
    ./out_fase1/plan_estructuras_full.parquet
    ./out_fase1/plan_sismos.parquet
    ./out_fase1/plan_experimentos.parquet

Uso:
    python inspect_plan.py

Salida:
    - Texto en consola con resúmenes por sección.
    - Plots adicionales en ./out_inspect/
================================================================================
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PLAN_DIR = Path("./out_fase1")
OUT_DIR = Path("./out_inspect")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def cargar_plan():
    estr = pd.read_parquet(PLAN_DIR / "plan_estructuras.parquet")
    estr_full = pd.read_parquet(PLAN_DIR / "plan_estructuras_full.parquet")
    sismos = pd.read_parquet(PLAN_DIR / "plan_sismos.parquet")
    plan = pd.read_parquet(PLAN_DIR / "plan_experimentos.parquet")
    return estr, estr_full, sismos, plan


def resumen_global(estr, sismos, plan):
    print("=" * 70)
    print("RESUMEN GLOBAL DEL PLAN")
    print("=" * 70)
    print(f"  Estructuras únicas:            {len(estr):,}")
    print(f"  Sismos:                        {len(sismos)}")
    n_configs = plan.groupby('estructura_id')['config_local_id'].nunique().sum()
    print(f"  Configs totales:               {n_configs:,}")
    print(f"  Filas (simulaciones):          {len(plan):,}")
    print(f"  Configs promedio/estructura:   "
          f"{plan.groupby('estructura_id')['config_local_id'].nunique().mean():.1f}")
    print(f"  Configs mín/máx por estr.:     "
          f"{plan.groupby('estructura_id')['config_local_id'].nunique().min()} / "
          f"{plan.groupby('estructura_id')['config_local_id'].nunique().max()}")

    n_sims = len(plan)
    print(f"\n  Tamaño esperado de resultados: ~{n_sims*220/1e6:.0f} MB en parquet")


def estadisticas_estructuras(estr):
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS — VARIABLES DE ESTRUCTURA")
    print("=" * 70)
    print("\n  Variables continuas:")
    cont = estr[["T1_objetivo", "T1_real", "T1_error_pct",
                 "masa_promedio_ton", "masa_total_ton", "k_promedio_Nm"]]
    print(cont.describe().round(4).to_string())

    print("\n  n_pisos:")
    print(estr["n_pisos"].value_counts().sort_index().to_string())

    print("\n  perfil_masa:")
    print(estr["perfil_masa"].value_counts().to_string())

    print("\n  perfil_rigidez:")
    print(estr["perfil_rigidez"].value_counts().to_string())


def estadisticas_configs(plan):
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS — CONFIGURACIONES DE AMORTIGUADORES")
    print("=" * 70)
    print(plan[["Mf_KN", "N_Dampers", "densidad_dampers"]].describe().round(3).to_string())

    print("\n  N_Dampers por n_pisos (mediana ± std):")
    grp = plan.groupby("n_pisos")["N_Dampers"].agg(["min", "median", "max", "mean", "std"])
    print(grp.round(2).to_string())


def estadisticas_sismos(sismos):
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS — SISMOS")
    print("=" * 70)
    if sismos["archivo"].notna().any():
        print(sismos.drop(columns=["archivo"]).to_string(index=False))
    else:
        print("  [WARN] Ningún sismo cargado todavía. Recalcula Fase 1 con SISMOS_DIR válido.")
        print(f"     Sismos esperados: {sismos['sismo'].tolist()}")


def verificar_consistencia(estr_full, plan):
    print("\n" + "=" * 70)
    print("VERIFICACIONES DE CONSISTENCIA")
    print("=" * 70)
    issues = []

    err_max = estr_full["T1_error_pct"].max()
    if err_max < 0.01:
        print(f"  [OK]  Calibración T1: error máx {err_max:.6f}% (excelente)")
    elif err_max < 0.5:
        print(f"  [WARN] Calibración T1: error máx {err_max:.4f}% (aceptable)")
    else:
        print(f"  [ERR] Calibración T1: error máx {err_max:.4f}% (revisar)")
        issues.append("T1_error_alto")

    bad_m = bad_k = 0
    for _, r in estr_full.iterrows():
        m = json.loads(r["m_vec_kg"])
        k = json.loads(r["k_vec_Nm"])
        if len(m) != r["n_pisos"]: bad_m += 1
        if len(k) != r["n_pisos"]: bad_k += 1
    if bad_m == 0 and bad_k == 0:
        print(f"  [OK]  Vectores m_vec y k_vec consistentes en {len(estr_full)} estructuras")
    else:
        print(f"  [ERR] {bad_m} con m_vec mal, {bad_k} con k_vec mal")
        issues.append("vectores_inconsistentes")

    plan_check = plan.copy()
    plan_check["bits_len"] = plan_check["Bits"].str.len()
    inconsistentes = (plan_check["bits_len"] != plan_check["n_pisos"]).sum()
    if inconsistentes == 0:
        print(f"  [OK]  Bits con longitud == n_pisos en todas las filas")
    else:
        print(f"  [ERR] {inconsistentes} filas con Bits de longitud incorrecta")
        issues.append("bits_longitud_mala")

    sin_dampers = (plan["N_Dampers"] == 0).sum()
    if sin_dampers == 0:
        print(f"  [OK]  Todas las configuraciones tienen al menos 1 amortiguador")
    else:
        print(f"  [WARN] {sin_dampers} configuraciones con 0 amortiguadores")

    mf_fuera = ((plan["Mf_KN"] < 25) | (plan["Mf_KN"] > 500)).sum()
    if mf_fuera == 0:
        print(f"  [OK]  Todos los Mf en [25, 500] KN")
    else:
        print(f"  [ERR] {mf_fuera} filas con Mf fuera de rango")
        issues.append("mf_fuera_rango")

    estr_cov = plan.groupby(["perfil_masa", "perfil_rigidez"])["estructura_id"].nunique()
    print(f"\n  Estructuras únicas por combinación (perfil_masa × perfil_rigidez):")
    print(estr_cov.unstack().to_string())

    if not issues:
        print("\n  [OK] Plan listo para Fase 2")
    else:
        print(f"\n  [ERR] {len(issues)} problemas detectados: {issues}")
    return issues


def mostrar_ejemplos(estr_full, plan):
    print("\n" + "=" * 70)
    print("EJEMPLOS — 3 ESTRUCTURAS ALEATORIAS")
    print("=" * 70)
    sample_ids = estr_full.sample(3, random_state=0)["estructura_id"].tolist()
    for eid in sample_ids:
        e = estr_full[estr_full["estructura_id"] == eid].iloc[0]
        m = json.loads(e["m_vec_kg"])
        k = json.loads(e["k_vec_Nm"])
        print(f"\n  Estructura #{eid}:")
        print(f"     n_pisos       = {e['n_pisos']}")
        print(f"     T1_objetivo   = {e['T1_objetivo']:.3f} s")
        print(f"     T1_real       = {e['T1_real']:.3f} s  (err {e['T1_error_pct']:.4f}%)")
        print(f"     masa promedio = {e['masa_promedio_ton']:.1f} ton/piso")
        print(f"     perfil_masa   = {e['perfil_masa']}")
        print(f"     perfil_rigidez= {e['perfil_rigidez']}")
        print(f"     m_vec [ton]   = {[round(x/1000, 1) for x in m]}")
        print(f"     k_vec [MN/m]  = {[round(x/1e6, 2) for x in k]}")
        configs = plan[plan["estructura_id"] == eid][["Bits", "N_Dampers", "Mf_KN"]].drop_duplicates()
        print(f"     Configs únicas: {len(configs)}")
        print(f"     Ejemplos Bits: {configs['Bits'].head(5).tolist()}")


def plots_adicionales(estr, plan, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for perfil in estr["perfil_rigidez"].unique():
        sub = estr[estr["perfil_rigidez"] == perfil]
        ax.scatter(sub["T1_objetivo"], sub["n_pisos"], alpha=0.5, s=20, label=perfil)
    ax.set_xlabel("T1 [s]"); ax.set_ylabel("n_pisos")
    ax.set_title("T1 vs n_pisos por perfil de rigidez")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "T1_vs_npisos_por_perfil_k.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(estr["T1_real"], estr["k_promedio_Nm"] / 1e6,
                    c=estr["n_pisos"], cmap="viridis", alpha=0.5, s=15)
    ax.set_xlabel("T1 real [s]"); ax.set_ylabel("k promedio [MN/m]")
    ax.set_title("Rigidez calibrada vs T1 (color = n_pisos)")
    ax.set_yscale("log")
    plt.colorbar(sc, ax=ax, label="n_pisos")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(out_dir / "k_vs_T1.png", dpi=120); plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, npiso in zip(axes.flat, sorted(plan["n_pisos"].unique())):
        sub = plan[plan["n_pisos"] == npiso]
        ax.hist(sub["N_Dampers"], bins=np.arange(0, npiso + 2) - 0.5, edgecolor="k")
        ax.set_title(f"n_pisos = {npiso}"); ax.set_xlabel("N_Dampers")
        ax.grid(alpha=0.3)
    plt.suptitle("N_Dampers por altura del edificio", y=1.02)
    plt.tight_layout(); plt.savefig(out_dir / "Ndampers_por_altura.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    h, xedges, yedges = np.histogram2d(estr["T1_objetivo"], estr["masa_promedio_ton"], bins=20)
    im = ax.imshow(h.T, origin="lower", aspect="auto",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap="YlOrRd")
    ax.set_xlabel("T1 [s]"); ax.set_ylabel("Masa promedio [ton]")
    ax.set_title("Densidad en espacio T1 × masa")
    plt.colorbar(im, ax=ax, label="N estructuras")
    plt.tight_layout(); plt.savefig(out_dir / "heatmap_T1_masa.png", dpi=120); plt.close()

    print(f"\n  [OK] Plots guardados en: {out_dir.resolve()}")


def main():
    print("Cargando plan...\n")
    estr, estr_full, sismos, plan = cargar_plan()
    resumen_global(estr, sismos, plan)
    estadisticas_estructuras(estr)
    estadisticas_configs(plan)
    estadisticas_sismos(sismos)
    issues = verificar_consistencia(estr_full, plan)
    mostrar_ejemplos(estr_full, plan)
    plots_adicionales(estr, plan, OUT_DIR)
    print("\n" + "=" * 70)
    print("INSPECCIÓN COMPLETADA" if not issues else "INSPECCIÓN COMPLETADA CON ALERTAS")
    print("=" * 70)


if __name__ == "__main__":
    main()
