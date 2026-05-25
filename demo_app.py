"""
================================================================================
DEMO INTERACTIVO — Optimizador de Amortiguadores de Fricción Rotacionales
================================================================================
Interfaz gráfica Streamlit para proponer una estructura nueva y recibir
la configuración óptima de amortiguadores según 3 criterios de diseño.

Uso:
    streamlit run demo_app.py
================================================================================
"""

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.express as px
import scipy.linalg as sla
import streamlit as st
from pathlib import Path
from scipy.stats import qmc

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Optimizador de Amortiguadores de Fricción",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR   = Path("./out_ml/models")
DATASET_PATH = Path("./out_ml/dataset_ml.parquet")

TARGETS = {
    "CR2_median": ("Mediana",      "Escenario típico",                    "#2196F3"),
    "CR2_p10":    ("P10",          "Caso favorable (mejor desempeño)",    "#43A047"),
    "CR2_p90":    ("P90",          "Caso crítico — diseño conservador",   "#F44336"),
}
TARGET_SUFFIXES = {"CR2_median": "median", "CR2_p10": "p10", "CR2_p90": "p90"}

N_PISOS_VALUES   = [3, 5, 7, 10, 12, 15, 18, 20]
PERFILES_MASA    = ["constante", "azotea_reducida", "lineal_decreciente"]
PERFILES_RIGIDEZ = ["constante", "lineal_decreciente", "logaritmico", "bilineal"]
MASS_TON_TO_KG   = 1000.0
G_MS2            = 9.81
N_CONFIGS_APP    = 300   # configuraciones LHS generadas para el demo


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELOS Y DATASET (cacheado)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def cargar_modelos():
    models = {tc: {} for tc in TARGETS}
    if not MODELS_DIR.exists():
        return models, None
    for f in MODELS_DIR.glob("*.pkl"):
        for tc, suf in TARGET_SUFFIXES.items():
            if f.stem.endswith(f"_{suf}"):
                name = f.stem.replace(f"_{suf}", "")
                with open(f, "rb") as fp:
                    models[tc][name] = pickle.load(fp)
    best = {}
    for tc, mdict in models.items():
        for name in ["XGBoost", "RandomForest", "Ridge", "LinearRegression"]:
            if name in mdict:
                best[tc] = (name, mdict[name])
                break
    return models, best


@st.cache_data
def cargar_estadisticas_dataset():
    if not DATASET_PATH.exists():
        return {"energia_mean": 0.0, "energia_std": 1.0}
    df = pd.read_parquet(DATASET_PATH, columns=["energia_total_normalizada"])
    return {
        "energia_mean": float(df["energia_total_normalizada"].mean()),
        "energia_std":  float(df["energia_total_normalizada"].std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES ESTRUCTURALES (replicadas de Fase 1)
# ─────────────────────────────────────────────────────────────────────────────

def perfil_masa(N, tipo, masa_promedio):
    if tipo == "constante":
        m = np.ones(N)
    elif tipo == "azotea_reducida":
        m = np.ones(N); m[-1] = 0.7
    else:  # lineal_decreciente
        m = np.linspace(1.15, 0.85, N)
    m = m * (masa_promedio / m.mean())
    return m * MASS_TON_TO_KG


def perfil_rigidez(N, tipo):
    if tipo == "constante":
        return np.ones(N)
    elif tipo == "lineal_decreciente":
        return np.linspace(1.0, 0.4, N)
    elif tipo == "logaritmico":
        return np.exp(-1.5 * np.linspace(0, 1, N))
    else:  # bilineal
        return np.where(np.arange(N) < N // 2, 1.0, 0.5).astype(float)


def build_K_shear(k_vec):
    N = len(k_vec)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = k_vec[i] + (k_vec[i + 1] if i < N - 1 else 0.0)
        if i > 0: K[i, i - 1] = -k_vec[i]
        if i < N - 1: K[i, i + 1] = -k_vec[i + 1]
    return K


def calcular_T1(M, K):
    vals, _ = sla.eigh(K, M)
    return 2 * np.pi / np.sqrt(max(vals[0], 1e-12))


def calibrar_estructura(N, perfil_k_tipo, m_vec, T1_objetivo):
    k_rel  = perfil_rigidez(N, perfil_k_tipo)
    k_init = k_rel * 1e8
    M = np.diag(m_vec)
    K = build_K_shear(k_init)
    T1_act = calcular_T1(M, K)
    k_scale = (T1_act / T1_objetivo) ** 2
    k_vec = k_init * k_scale
    T1_real = calcular_T1(M, build_K_shear(k_vec))
    return k_vec, T1_real


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE CONFIGURACIONES DE AMORTIGUADORES
# ─────────────────────────────────────────────────────────────────────────────

def generar_configuraciones(N, Mf_min, Mf_max, n_lhs=N_CONFIGS_APP, seed=42):
    configs = []

    # Estructuradas: un amortiguador por piso, todos, mitad inf, mitad sup
    rng = np.random.default_rng(seed)
    for i in range(N):
        bits = np.zeros(N, dtype=int); bits[i] = 1
        configs.append((bits, float(rng.uniform(Mf_min, Mf_max))))
    for pattern in [np.ones(N, dtype=int),
                    np.array([1]*( N//2) + [0]*(N - N//2)),
                    np.array([0]*(N//2) + [1]*(N - N//2))]:
        configs.append((pattern, float(rng.uniform(Mf_min, Mf_max))))

    # LHS: densidad × Mf
    sampler = qmc.LatinHypercube(d=2, seed=seed, scramble=True)
    U = sampler.random(n=n_lhs)
    densidades = 0.1 + U[:, 0] * 0.9
    mfs = Mf_min + U[:, 1] * (Mf_max - Mf_min)
    rng2 = np.random.default_rng(seed + 1)
    for d, mf in zip(densidades, mfs):
        n_active = max(1, int(round(d * N)))
        idx = rng2.choice(N, size=n_active, replace=False)
        bits = np.zeros(N, dtype=int); bits[idx] = 1
        configs.append((bits, float(mf)))

    # Deduplicar por (bits_str, round(Mf,1))
    seen, unique = set(), []
    for bits, mf in configs:
        key = ("".join(map(str, bits)), round(mf, 1))
        if key not in seen:
            seen.add(key); unique.append((bits, mf))
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def construir_features(N, T1_real, configs, energia_mean, max_pisos=20):
    rows = []
    for bits, mf in configs:
        n_dam = int(bits.sum())
        positions = [i + 1 for i, b in enumerate(bits) if b == 1]
        pos_mean  = np.mean(positions) if positions else 0.0
        pos_norm  = pos_mean / N
        concentr  = (np.std(positions) / N) if len(positions) > 1 else 0.0

        row = {
            "n_pisos_first":              N,
            "T1_real_first":              T1_real,
            "Mf_KN_first":               mf,
            "N_Dampers_first":           n_dam,
            "densidad_dampers":          n_dam / N,
            "posicion_promedio_dampers": pos_mean,
            "posicion_normalizada":      pos_norm,
            "concentracion_dampers":     concentr,
            "energia_total_normalizada": energia_mean,
            "Bits_first":                "".join(map(str, bits)),
        }
        for i in range(max_pisos):
            row[f"damper_piso{i+1}"] = int(bits[i]) if i < N else 0
        rows.append(row)
    return pd.DataFrame(rows)


def feature_cols_from_df(df, max_pisos=20):
    base = ["n_pisos_first", "T1_real_first", "Mf_KN_first", "N_Dampers_first",
            "densidad_dampers", "posicion_promedio_dampers", "posicion_normalizada",
            "concentracion_dampers", "energia_total_normalizada"]
    damper_cols = [f"damper_piso{i+1}" for i in range(max_pisos)
                   if f"damper_piso{i+1}" in df.columns]
    return base + damper_cols


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def fig_edificio(N, bits_str, titulo="", cr2=None):
    bits = [int(b) for b in bits_str]
    fig, ax = plt.subplots(figsize=(3.2, N * 0.55 + 1.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, N + 1)
    ax.axis("off")

    ancho, h_piso = 6.0, 1.0
    x0 = 2.0

    fig.patch.set_facecolor("#FAFCFF")
    ax.set_facecolor("#FAFCFF")

    for piso in range(N):
        y = piso * h_piso
        color = "#1565C0" if bits[piso] else "#EEF4FF"
        borde = "#0D47A1" if bits[piso] else "#BBDEFB"
        lw    = 2.0 if bits[piso] else 0.8

        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y + 0.05), ancho, h_piso * 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=borde, linewidth=lw, zorder=2))

        ax.text(x0 - 0.3, y + h_piso * 0.5, str(piso + 1),
                ha="right", va="center", fontsize=8, color="#546E7A")

        if bits[piso]:
            ax.text(x0 + ancho / 2, y + h_piso * 0.5,
                    "  ⬡ amortiguador", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=3)

    # Suelo
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 - 0.3, -0.15), ancho + 0.6, 0.2,
        boxstyle="square,pad=0", facecolor="#455A64", zorder=2))

    titulo_txt = titulo
    if cr2 is not None:
        titulo_txt += f"\nCR2 = {cr2:.3f}"
    ax.set_title(titulo_txt, fontsize=9, fontweight="bold", pad=4)
    plt.tight_layout(pad=0.3)
    return fig


def fig_pareto_plotly(df_configs):
    fig = go.Figure()

    # Fondo: todas las configuraciones (mediana real si existe, sino predicha)
    y_col = "CR2_median" if "CR2_median" in df_configs.columns else "pred_CR2_median"
    fig.add_trace(go.Scatter(
        x=df_configs["N_Dampers_first"],
        y=df_configs[y_col],
        mode="markers",
        marker=dict(color="lightgray", size=5, opacity=0.5),
        name="Todas las configs",
        hovertemplate="N_Dam=%{x}<br>CR2=%{y:.3f}<extra></extra>",
    ))

    colores = {"CR2_median": "#2196F3", "CR2_p10": "#F44336", "CR2_p90": "#FF9800"}
    for tc, (short, desc, color) in TARGETS.items():
        pcol = f"pred_{tc}"
        if pcol not in df_configs.columns:
            continue
        top5 = df_configs.nsmallest(5, pcol)
        fig.add_trace(go.Scatter(
            x=top5["N_Dampers_first"],
            y=top5[pcol],
            mode="markers+text",
            marker=dict(color=color, size=14, symbol="star",
                        line=dict(color="black", width=1)),
            text=[f"#{i+1}" for i in range(len(top5))],
            textposition="top center",
            textfont=dict(size=9),
            name=f"Top-5 {short}",
            hovertemplate=(
                f"<b>{short}</b><br>N_Dam=%{{x}}<br>"
                f"CR2_pred=%{{y:.3f}}<br>"
                f"Bits=%{{customdata}}<extra></extra>"
            ),
            customdata=top5["Bits_first"].values,
        ))

    fig.update_layout(
        title=dict(text="Pareto Front — Desempeño vs Costo",
                   font=dict(size=14, family="Inter, Arial", color="#1A1A2E")),
        xaxis_title="N° de Amortiguadores (costo)",
        yaxis_title="CR2 predicho (menor = mejor)",
        legend=dict(orientation="h", y=-0.20,
                    font=dict(size=11, color="#546E7A")),
        height=430,
        margin=dict(l=55, r=30, t=60, b=90),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#E8EDF4",
                   tickfont=dict(color="#546E7A"),
                   title_font=dict(color="#546E7A")),
        yaxis=dict(showgrid=True, gridcolor="#E8EDF4", autorange="reversed",
                   tickfont=dict(color="#546E7A"),
                   title_font=dict(color="#546E7A")),
        hovermode="closest",
        font=dict(family="Inter, Arial"),
    )
    return fig


def fig_radar_reduccion(cr2_vals):
    """Gráfico de barras mostrando CR2 por target vs benchmark CR2=1."""
    targets_short = ["Mediana", "P10\n(peor caso)", "P90\n(conservador)"]
    values  = list(cr2_vals.values())
    colors  = ["#1565C0", "#C62828", "#E65100"]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    bars = ax.barh(targets_short, values, color=colors, alpha=0.88, height=0.48)
    ax.axvline(1.0, color="#90A4AE", linestyle="--", lw=1.2,
               label="Sin amortiguador (CR2=1)")
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold",
                color="#1A1A2E")

    reduccion = [(1 - v) * 100 for v in values]
    for r, bar in zip(reduccion, bars):
        if r > 0:
            ax.text(0.02, bar.get_y() + bar.get_height() / 2,
                    f"−{r:.1f}%", va="center", fontsize=8.5, color="white",
                    fontweight="bold")

    ax.set_xlim(0, max(max(values) * 1.15, 1.15))
    ax.set_xlabel("CR2 (Combined Response Ratio)", fontsize=9, color="#546E7A")
    ax.set_title("Desempeño del diseño recomendado", fontsize=10,
                 fontweight="bold", color="#1A1A2E", pad=8)
    ax.legend(fontsize=8)
    ax.tick_params(colors="#546E7A")
    for spine in ax.spines.values():
        spine.set_edgecolor("#E0E7EF")
    ax.grid(alpha=0.4, axis="x", color="#E0E7EF")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── CSS — tema claro con contraste forzado ────────────────────────────
    st.markdown("""
    <style>
    /* ── Fondo general ── */
    .stApp, .main, .block-container {
        background-color: #F4F7FB !important;
        color: #1A1A2E !important;
    }

    /* ── Texto global ── */
    .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp li, .stApp td, .stApp th {
        color: #1A1A2E !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: #0D1B2A !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #E8EEF6 !important;
        border-right: 1px solid #C8D4E8;
    }
    [data-testid="stSidebar"] * { color: #1A1A2E !important; }
    [data-testid="stSidebar"] h3 {
        color: #1565C0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Labels de sliders y selects ── */
    .stSlider label, .stSelectbox label, .stCheckbox label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #2C3E50 !important;
        font-weight: 500 !important;
    }

    /* ── Valores de sliders ── */
    .stSlider [data-testid="stThumbValue"],
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #1565C0 !important;
    }

    /* ── Selectbox y opciones ── */
    [data-baseweb="select"] div,
    [data-baseweb="select"] span {
        color: #1A1A2E !important;
        background-color: #FFFFFF !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
        border: 1px solid #D0DAF0;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] p {
        color: #4A5568 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0D1B2A !important;
    }
    [data-testid="stMetricDelta"] {
        color: #2E7D32 !important;
    }

    /* ── Captions ── */
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #4A5568 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #DDE6F4 !important;
        border-radius: 8px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #2C3E50 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1565C0 !important;
        font-weight: 600 !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrameResizable"] * {
        color: #1A1A2E !important;
        background-color: #FFFFFF !important;
    }

    /* ── Alertas / info / warning ── */
    div[data-testid="stInfo"] {
        background: #DBEAFE !important;
        border-left-color: #1565C0 !important;
    }
    div[data-testid="stInfo"] * { color: #1E3A5F !important; }

    div[data-testid="stSuccess"] {
        background: #DCFCE7 !important;
    }
    div[data-testid="stSuccess"] * { color: #14532D !important; }

    div[data-testid="stWarning"] {
        background: #FEF9C3 !important;
    }
    div[data-testid="stWarning"] * { color: #713F12 !important; }

    div[data-testid="stError"] {
        background: #FEE2E2 !important;
    }
    div[data-testid="stError"] * { color: #7F1D1D !important; }

    /* ── Expander ── */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #C8D4E8 !important;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] summary p {
        color: #1A1A2E !important;
        font-weight: 600 !important;
    }

    /* ── Botón primario ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1565C0, #1976D2) !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(21,101,192,0.35);
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 14px rgba(21,101,192,0.50) !important;
        transform: translateY(-1px);
    }

    /* ── Botón download ── */
    .stDownloadButton > button {
        color: #1565C0 !important;
        border-color: #1565C0 !important;
        font-weight: 600;
    }

    /* ── Divisores ── */
    hr { border-color: #C8D4E8 !important; }

    /* ── Spinner ── */
    [data-testid="stSpinner"] p { color: #1565C0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#FFFFFF; border-left:6px solid #1565C0;
                box-shadow:0 2px 14px rgba(0,0,0,0.07);
                padding:1.4rem 2rem; border-radius:12px; margin-bottom:1.5rem;'>
        <div style='display:flex; align-items:center; gap:0.9rem;'>
            <span style='font-size:2.2rem; line-height:1;'>🏗️</span>
            <div>
                <h1 style='color:#1A1A2E; margin:0; font-size:1.55rem; font-weight:700;
                           letter-spacing:-0.01em;'>
                    Optimizador de Amortiguadores de Fricción Rotacionales
                </h1>
                <p style='color:#546E7A; margin:0.25rem 0 0; font-size:0.9rem;'>
                    Sistema inteligente basado en Machine Learning para el diseño sísmico óptimo
                </p>
            </div>
        </div>
        <div style='display:flex; gap:1.5rem; margin-top:0.9rem;
                    padding-top:0.9rem; border-top:1px solid #EBF0F8; flex-wrap:wrap;'>
            <span style='color:#1565C0; font-size:0.8rem; font-weight:600;'>
                ⚡ ML-powered
            </span>
            <span style='color:#78909C; font-size:0.8rem;'>
                ~440,000 simulaciones dinámicas
            </span>
            <span style='color:#78909C; font-size:0.8rem;'>
                NSD-LCP-PSOR · Marcos de acero
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cargar recursos ──────────────────────────────────────────────────────
    _, best_models = cargar_modelos()
    stats = cargar_estadisticas_dataset()

    if best_models is None or len(best_models) == 0:
        st.error("No se encontraron modelos entrenados. Corre `python 04_full_pipeline.py` primero.")
        st.stop()

    targets_disponibles = list(best_models.keys())

    # ── Sidebar — Parámetros de la estructura ────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Parámetros de la Estructura")
        st.divider()

        n_pisos = st.selectbox("Número de pisos", N_PISOS_VALUES, index=3)
        T1 = st.slider("Período fundamental T₁ [s]", 0.3, 3.0, 1.2, 0.05,
                       help="Período del primer modo de vibración")
        masa = st.slider("Masa promedio por piso [ton]", 150, 350, 250, 10)
        perf_masa = st.selectbox("Perfil de masa", PERFILES_MASA)
        perf_rig  = st.selectbox("Perfil de rigidez", PERFILES_RIGIDEZ)

        st.divider()
        st.markdown("### 🔩 Parámetros de Amortiguadores")
        Mf_min = st.slider("Mf mínimo [kN]", 25, 200, 50, 25)
        Mf_max = st.slider("Mf máximo [kN]", 100, 500, 400, 25)
        if Mf_max <= Mf_min:
            st.warning("Mf máximo debe ser mayor que mínimo.")
            Mf_max = Mf_min + 50

        st.divider()
        st.markdown("### 🎯 Diseño por mínimo de amortiguadores")
        usar_umbral = st.checkbox("Activar restricción de CR2 objetivo", value=True)
        cr2_umbral = st.slider(
            "CR2 máximo aceptable", 0.50, 1.00, 0.85, 0.01,
            help="Busca la config con MENOS amortiguadores que aún cumpla CR2 ≤ este valor",
            disabled=not usar_umbral,
        )
        tc_umbral = st.selectbox(
            "Target para el umbral",
            options=list(TARGETS.keys()),
            format_func=lambda k: TARGETS[k][0],
            disabled=not usar_umbral,
        )

        st.divider()
        n_top = st.slider("Top-N configuraciones a mostrar", 3, 10, 5)
        calcular = st.button("🔍 Optimizar configuración", type="primary",
                             use_container_width=True)

        st.divider()
        st.caption(f"Modelos disponibles: {', '.join(targets_disponibles)}")
        model_names = {tc: nm for tc, (nm, _) in best_models.items()}
        for tc, nm in model_names.items():
            st.caption(f"• {tc}: **{nm}**")

    # ── Resultado ────────────────────────────────────────────────────────────
    if not calcular:
        st.info("👈 Configura los parámetros en el panel izquierdo y presiona **Optimizar**.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estructuras en base de datos", "1,000")
        with col2:
            st.metric("Simulaciones realizadas", "~440,000")
        with col3:
            st.metric("Modelos entrenados", f"{len(targets_disponibles)} targets")
        return

    # ── Cálculo ──────────────────────────────────────────────────────────────
    with st.spinner("Calibrando estructura y generando configuraciones..."):
        m_vec = perfil_masa(n_pisos, perf_masa, masa)
        k_vec, T1_real = calibrar_estructura(n_pisos, perf_rig, m_vec, T1)
        T1_error = abs(T1_real - T1) / T1 * 100

        configs = generar_configuraciones(n_pisos, Mf_min, Mf_max,
                                          n_lhs=N_CONFIGS_APP)
        df_feat = construir_features(n_pisos, T1_real, configs,
                                     stats["energia_mean"])
        fcols = feature_cols_from_df(df_feat)
        X = df_feat[fcols].values

    with st.spinner("Prediciendo con modelos ML..."):
        for tc in targets_disponibles:
            _, model = best_models[tc]
            df_feat[f"pred_{tc}"] = model.predict(X)

    # ── Resumen de la estructura ─────────────────────────────────────────────
    st.markdown("#### 📐 Estructura calibrada")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pisos", n_pisos)
    c2.metric("T₁ real [s]", f"{T1_real:.3f}")
    c3.metric("Error T₁", f"{T1_error:.3f}%")
    c4.metric("Masa total [ton]", f"{m_vec.sum()/1000:.0f}")
    c5.metric("Configs evaluadas", f"{len(df_feat):,}")
    st.divider()

    # ── Mejor configuración por target ──────────────────────────────────────
    mejores = {}
    for tc in targets_disponibles:
        mejores[tc] = df_feat.nsmallest(1, f"pred_{tc}").iloc[0]

    # Métricas principales
    st.markdown("#### 🏆 Diseño óptimo por criterio")
    cols = st.columns(len(targets_disponibles))
    for col, tc in zip(cols, targets_disponibles):
        short, desc, color = TARGETS[tc]
        row = mejores[tc]
        cr2_val = float(row[f"pred_{tc}"])
        nd      = int(row["N_Dampers_first"])
        mf      = float(row["Mf_KN_first"])
        delta   = (1 - cr2_val) * 100
        col.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E0E7EF;
                    border-top: 4px solid {color};
                    border-radius:12px; padding:1rem 1.2rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
            <div style='display:flex; align-items:center;
                        justify-content:space-between; margin-bottom:0.4rem;'>
                <span style='color:{color}; font-weight:700; font-size:0.92rem;'>
                    {short}
                </span>
                <span style='background:{color}15; color:{color}; font-size:0.7rem;
                             font-weight:600; padding:2px 8px; border-radius:20px;'>
                    {desc.split("(")[0].strip()}
                </span>
            </div>
            <div style='font-size:2.1rem; font-weight:800; color:#1A1A2E;
                        letter-spacing:-0.02em;'>
                {cr2_val:.3f}
            </div>
            <div style='color:#2E7D32; font-size:0.83rem; font-weight:600;
                        margin:0.25rem 0 0.6rem;'>
                ▼ {delta:.1f}% reducción
            </div>
            <div style='border-top:1px solid #F0F4FA; padding-top:0.5rem;
                        font-size:0.8rem; color:#546E7A;'>
                🔩 {nd} amortiguadores &nbsp;·&nbsp; Mf = {mf:.0f} kN
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Convergencia ─────────────────────────────────────────────────────────
    bits_tops = {tc: mejores[tc]["Bits_first"] for tc in targets_disponibles}
    converge  = len(set(bits_tops.values())) == 1
    nd_tops   = {tc: int(mejores[tc]["N_Dampers_first"]) for tc in targets_disponibles}
    nd_iguales = len(set(nd_tops.values())) == 1

    if converge:
        st.success("✅ **Los 3 criterios convergen en el mismo diseño.** "
                   "Alta confianza en la recomendación.")
    elif nd_iguales:
        st.warning("⚠️ Mismo número de amortiguadores, distribución distinta. "
                   "Los criterios son consistentes en costo, difieren en ubicación.")
    else:
        st.info("ℹ️ Los 3 criterios recomiendan configuraciones distintas. "
                "Ver análisis Pareto para seleccionar según nivel de riesgo.")

    st.divider()

    # ── Mínimo de amortiguadores ─────────────────────────────────────────────
    st.markdown("#### 🎯 Diseño de mínimo costo: menos amortiguadores que cumplan CR2 ≤ umbral")

    if usar_umbral and tc_umbral in targets_disponibles:
        pred_col = f"pred_{tc_umbral}"
        factibles = df_feat[df_feat[pred_col] <= cr2_umbral].copy()
        short_tc, _, _ = TARGETS[tc_umbral]

        if len(factibles) == 0:
            cr2_min = float(df_feat[pred_col].min())
            st.error(
                f"Ninguna configuración alcanza CR2 ≤ {cr2_umbral} "
                f"para **{short_tc}**. "
                f"Mejor CR2 alcanzable: **{cr2_min:.3f}**. "
                f"Sube el umbral a ≥ {cr2_min + 0.02:.2f} o amplía el rango de Mf."
            )
        else:
            min_row = factibles.nsmallest(1, "N_Dampers_first").iloc[0]
            nd_min  = int(min_row["N_Dampers_first"])
            mf_min  = float(min_row["Mf_KN_first"])
            cr2_min_val = float(min_row[pred_col])
            bits_min = min_row["Bits_first"]
            nd_optimo = int(mejores[tc_umbral]["N_Dampers_first"])
            ahorro = nd_optimo - nd_min

            # Métricas
            ca, cb, cc, cd = st.columns(4)
            ca.metric("Amortiguadores mínimos", nd_min,
                      delta=f"−{ahorro} vs diseño óptimo" if ahorro > 0 else "= diseño óptimo",
                      delta_color="inverse")
            cb.metric(f"CR2 ({short_tc})", f"{cr2_min_val:.3f}",
                      delta=f"umbral = {cr2_umbral}", delta_color="off")
            cc.metric("Mf recomendado [kN]", f"{mf_min:.0f}")
            cd.metric("Configs factibles", f"{len(factibles):,}")

            # Diagrama + info
            col_ed, col_info = st.columns([1.4, 2.6], gap="large")
            with col_ed:
                fig_min = fig_edificio(
                    n_pisos, bits_min,
                    titulo=f"Mínimo costo ({short_tc})",
                    cr2=cr2_min_val,
                )
                st.pyplot(fig_min, use_container_width=False)
                plt.close(fig_min)

            with col_info:
                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #C8E6C9;
                            border-left:5px solid #2E7D32;
                            border-radius:12px; padding:1rem 1.3rem; margin-top:0.5rem;
                            box-shadow:0 2px 8px rgba(46,125,50,0.08);'>
                    <div style='color:#1B5E20; font-weight:700; font-size:0.95rem;
                                margin-bottom:0.6rem;'>
                        ✅ Configuración de mínimo costo encontrada
                    </div>
                    <table style='width:100%; font-size:0.87rem; border-collapse:collapse;'>
                        <tr>
                            <td style='color:#546E7A; padding:2px 0; width:52%;'>
                                Configuración (bits)
                            </td>
                            <td style='color:#1A1A2E; font-family:monospace;
                                       font-weight:600;'>
                                {bits_min}
                            </td>
                        </tr>
                        <tr>
                            <td style='color:#546E7A; padding:2px 0;'>
                                N° amortiguadores
                            </td>
                            <td style='color:#1A1A2E; font-weight:600;'>
                                {nd_min} / {n_pisos} pisos
                                &nbsp;<span style='color:#78909C;font-weight:400;'>
                                    ({nd_min/n_pisos*100:.0f}% densidad)
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td style='color:#546E7A; padding:2px 0;'>
                                Fuerza de deslizamiento
                            </td>
                            <td style='color:#1A1A2E; font-weight:600;'>
                                {mf_min:.0f} kN
                            </td>
                        </tr>
                        <tr>
                            <td style='color:#546E7A; padding:2px 0;'>
                                CR2 predicho ({short_tc})
                            </td>
                            <td style='color:#1A1A2E; font-weight:600;'>
                                {cr2_min_val:.4f}
                            </td>
                        </tr>
                        <tr>
                            <td style='color:#546E7A; padding:2px 0;'>
                                Reducción de respuesta
                            </td>
                            <td style='color:#2E7D32; font-weight:700;'>
                                {(1-cr2_min_val)*100:.1f}%
                            </td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                # Comparativa óptimo vs mínimo costo
                st.markdown("**Comparativa: óptimo (mejor CR2) vs mínimo costo**")
                opt_row = mejores[tc_umbral]
                comp_df = pd.DataFrame({
                    "Criterio":          ["Mínimo costo", "Óptimo (mejor CR2)"],
                    "N° Amortiguadores": [nd_min, nd_optimo],
                    "Mf [kN]":           [round(mf_min, 1),
                                          round(float(opt_row["Mf_KN_first"]), 1)],
                    "CR2 predicho":      [round(cr2_min_val, 4),
                                          round(float(opt_row[pred_col]), 4)],
                    "Reducción [%]":     [round((1-cr2_min_val)*100, 1),
                                          round((1-float(opt_row[pred_col]))*100, 1)],
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

    else:
        st.caption("Activa la restricción de CR2 en el panel lateral para ver este análisis.")

    st.divider()

    # ── Diagrama de edificios + Pareto + Reducción ───────────────────────────
    left, right = st.columns([2.5, 3.5], gap="large")

    with left:
        st.markdown("#### 🏢 Ubicación de amortiguadores")
        tab_names = [TARGETS[tc][0] for tc in targets_disponibles]
        tabs = st.tabs(tab_names)
        for tab, tc in zip(tabs, targets_disponibles):
            with tab:
                short, desc, color = TARGETS[tc]
                row  = mejores[tc]
                bits = row["Bits_first"]
                cr2  = float(row[f"pred_{tc}"])
                fig_ed = fig_edificio(n_pisos, bits, titulo=short, cr2=cr2)
                st.pyplot(fig_ed, use_container_width=False)
                plt.close(fig_ed)

    with right:
        st.markdown("#### 📊 Pareto Front interactivo")
        fig_p = fig_pareto_plotly(df_feat)
        st.plotly_chart(fig_p, use_container_width=True)

        # Gráfico de reducción
        cr2_vals = {tc: float(mejores[tc][f"pred_{tc}"]) for tc in targets_disponibles}
        fig_r = fig_radar_reduccion(cr2_vals)
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    st.divider()

    # ── Tabla top-N por target ───────────────────────────────────────────────
    st.markdown(f"#### 📋 Top-{n_top} configuraciones recomendadas")
    tab_names2 = [TARGETS[tc][0] + f" ({TARGETS[tc][1]})" for tc in targets_disponibles]
    tabs2 = st.tabs(tab_names2)
    for tab, tc in zip(tabs2, targets_disponibles):
        with tab:
            top = df_feat.nsmallest(n_top, f"pred_{tc}")[
                ["Bits_first", "N_Dampers_first", "Mf_KN_first",
                 "densidad_dampers", "posicion_normalizada", f"pred_{tc}"]
            ].copy()
            top.columns = ["Configuración (Bits)", "N° Amortiguadores",
                           "Mf [kN]", "Densidad", "Pos. promedio norm.", "CR2 predicho"]
            top = top.reset_index(drop=True)
            top.index += 1
            top["CR2 predicho"] = top["CR2 predicho"].round(4)
            top["Mf [kN]"] = top["Mf [kN]"].round(1)
            top["Densidad"] = top["Densidad"].round(3)
            top["Pos. promedio norm."] = top["Pos. promedio norm."].round(3)
            st.dataframe(top, use_container_width=True)

    st.divider()

    # ── Exportar ─────────────────────────────────────────────────────────────
    with st.expander("💾 Exportar resultados"):
        frames = []
        for tc in targets_disponibles:
            top_exp = df_feat.nsmallest(n_top, f"pred_{tc}")[
                ["Bits_first", "N_Dampers_first", "Mf_KN_first", f"pred_{tc}"]
            ].copy()
            top_exp.columns = ["Bits", "N_Dampers", "Mf_KN", "CR2_pred"]
            top_exp.insert(0, "target", tc)
            top_exp.insert(0, "n_pisos", n_pisos)
            top_exp.insert(0, "T1_real", round(T1_real, 4))
            frames.append(top_exp)
        export_df = pd.concat(frames, ignore_index=True)
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar recomendaciones CSV",
            data=csv,
            file_name=f"recomendacion_{n_pisos}pisos_T1{T1:.2f}.csv",
            mime="text/csv",
        )
        st.dataframe(export_df, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; margin-top:2.5rem; padding:1rem;
                border-top:1px solid #E0E7EF;'>
        <span style='color:#90A4AE; font-size:0.78rem;'>
            Optimizador ML · ~440,000 simulaciones dinámicas (NSD-LCP-PSOR)
            &nbsp;·&nbsp; Marcos de acero con amortiguadores de fricción rotacionales
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
