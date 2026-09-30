"""
================================================================================
FASE 4 — PIPELINE COMPLETO DE MACHINE LEARNING
================================================================================
Script automatizado que ejecuta todo el flujo de ML para TRES targets:
    - CR2_median : escenario esperado (mediana sobre los 8 sismos)
    - CR2_p10    : peor caso — el 10% más crítico de los sismos
    - CR2_p90    : escenario conservador — al menos el 90% de los sismos

Flujo por cada target:
    1. EDA (Exploratory Data Analysis) con plots
    2. Feature Engineering (compartido entre targets)
    3. Train-test split con GroupKFold por estructura
    4. Baseline models (Linear, Ridge)
    5. Ensemble models (Random Forest, XGBoost, LightGBM)
    6. Evaluación y comparación
    7. Feature importance analysis
    8. Guardado de modelos y resultados

Post-entrenamiento:
    9. Análisis de convergencia: ¿los 3 targets recomiendan el mismo diseño?

Uso:
    python 04_full_pipeline.py

Salidas:
    - out_ml/eda/               : plots exploratorios
    - out_ml/models/            : modelos entrenados (.pkl) con sufijo _median/_p10/_p90
    - out_ml/results/           : métricas y comparaciones por target
    - out_ml/dataset_ml.parquet : dataset procesado con features
    - out_ml/results/convergencia_targets.csv : análisis de convergencia
================================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# Configuración de paths
DATA_PATH = Path("./out_fase2/all_results.parquet")
OUT_DIR = Path("./out_ml")
EDA_DIR = OUT_DIR / "eda"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"

for d in [OUT_DIR, EDA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Los tres targets del experimento
TARGETS = {
    'CR2_median': 'Mediana — escenario típico',
    'CR2_p10':    'Percentil 10 — caso favorable (mejor desempeño)',
    'CR2_p90':    'Percentil 90 — caso crítico (diseño conservador)',
}

TARGET_COLORS = {
    'CR2_median': 'steelblue',
    'CR2_p10':    '#43A047',
    'CR2_p90':    'crimson',
}


# ========================================================================
# 1. CARGA Y AGREGACIÓN DE DATOS
# ========================================================================

def load_and_aggregate():
    print("=" * 70)
    print("FASE 4 — MACHINE LEARNING PIPELINE (3 targets: median / p10 / p90)")
    print("=" * 70)

    print(f"\n[1/8] Cargando dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"      Shape original: {df.shape}")

    print(f"\n[2/8] Agregando sobre sismos (mediana robusta + percentiles)...")
    agg = df.groupby(['estructura_id', 'config_local_id']).agg({
        'CR2': [
            'median',
            'mean',
            'std',
            lambda x: np.percentile(x, 10),   # peor caso
            lambda x: np.percentile(x, 90),   # escenario conservador
            'min',
            'max',
        ],
        'R1': 'median',
        'R2': 'median',
        'R3': 'median',
        'drift_ratio': 'median',
        'n_pisos': 'first',
        'T1_real': 'first',
        'Bits': 'first',
        'Mf_KN': 'first',
        'N_Dampers': 'first',
        'E_total_J': 'median',
    }).reset_index()

    agg.columns = [
        '_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col
        for col in agg.columns.values
    ]
    agg.rename(columns={
        'CR2_<lambda_0>': 'CR2_p10',
        'CR2_<lambda_1>': 'CR2_p90',
    }, inplace=True)

    print(f"      Shape agregado: {agg.shape}")
    print(f"      Estructuras únicas: {agg['estructura_id'].nunique()}")
    print(f"      Configs únicas: {len(agg)}")
    print(f"\n      Estadísticas por target:")
    for tc in ['CR2_p10', 'CR2_median', 'CR2_p90']:
        col = agg[tc]
        print(f"        {tc:12s}  media={col.mean():.3f}  std={col.std():.3f}  "
              f"min={col.min():.3f}  max={col.max():.3f}")

    return df, agg


# ========================================================================
# 2. EDA
# ========================================================================

def exploratory_analysis(df, agg):
    print(f"\n[3/8] EDA: generando plots en {EDA_DIR}/...")

    # Distribución de los 3 targets
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (tc, label) in zip(axes, TARGETS.items()):
        col = agg[tc]
        ax.hist(col, bins=50, edgecolor='k', alpha=0.7, color=TARGET_COLORS[tc])
        ax.axvline(col.median(), color='k', linestyle='--',
                   label=f'Mediana: {col.median():.3f}')
        ax.set_xlabel(f'CR2 ({tc})')
        ax.set_ylabel('Frecuencia')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EDA_DIR / '01_target_distribution_3targets.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Correlaciones de los 3 targets con N_Dampers y T1
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col_idx, (tc, label) in enumerate(TARGETS.items()):
        for row_idx, (feat, xlabel) in enumerate([
            ('N_Dampers_first', 'N_Dampers'),
            ('T1_real_first', 'T1 [s]'),
        ]):
            ax = axes[row_idx, col_idx]
            ax.scatter(agg[feat], agg[tc], alpha=0.2, s=4, color=TARGET_COLORS[tc])
            corr = agg[feat].corr(agg[tc])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(tc)
            ax.set_title(f'r = {corr:.3f}')
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EDA_DIR / '02_correlations_3targets.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Scatter p10 vs p90 (cuantifica la dispersión de cada config)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(agg['CR2_p10'], agg['CR2_p90'],
                    c=agg['N_Dampers_first'], cmap='viridis', alpha=0.3, s=5)
    lim = [agg[['CR2_p10', 'CR2_p90']].min().min(),
           agg[['CR2_p10', 'CR2_p90']].max().max()]
    ax.plot(lim, lim, 'r--', lw=1, label='p10 = p90 (sin dispersión)')
    ax.set_xlabel('CR2 p10 (peor caso)')
    ax.set_ylabel('CR2 p90 (conservador)')
    ax.set_title('Dispersión sísmica por configuración\n(coloreado por N_Dampers)')
    plt.colorbar(sc, ax=ax, label='N_Dampers')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EDA_DIR / '03_p10_vs_p90_dispersion.png', dpi=120, bbox_inches='tight')
    plt.close()

    print(f"      [OK] 3 plots generados en {EDA_DIR}/")

    best_per_structure = agg.loc[agg.groupby('estructura_id')['CR2_median'].idxmin()]
    return best_per_structure


# ========================================================================
# 3. FEATURE ENGINEERING
# ========================================================================

def feature_engineering(agg):
    print(f"\n[4/8] Feature Engineering...")

    max_pisos = agg['n_pisos_first'].max()
    for i in range(max_pisos):
        agg[f'damper_piso{i+1}'] = agg.apply(
            lambda row: int(row['Bits_first'][i]) if i < len(row['Bits_first']) else 0,
            axis=1
        )

    agg['densidad_dampers'] = agg['N_Dampers_first'] / agg['n_pisos_first']
    agg['Mf_por_damper'] = agg['Mf_KN_first']

    def calc_cm_dampers(row):
        bits = row['Bits_first']
        positions = [i + 1 for i, b in enumerate(bits) if b == '1']
        return np.mean(positions) if positions else 0

    def calc_concentracion(row):
        bits = row['Bits_first']
        positions = [i + 1 for i, b in enumerate(bits) if b == '1']
        if len(positions) <= 1:
            return 0
        return np.std(positions) / row['n_pisos_first']

    agg['posicion_promedio_dampers'] = agg.apply(calc_cm_dampers, axis=1)
    agg['posicion_normalizada'] = agg['posicion_promedio_dampers'] / agg['n_pisos_first']
    agg['concentracion_dampers'] = agg.apply(calc_concentracion, axis=1)

    dataset_path = OUT_DIR / "dataset_ml.parquet"
    agg.to_parquet(dataset_path, index=False)
    print(f"      [OK] Dataset guardado en: {dataset_path}")
    return agg


# ========================================================================
# 4. PREPARACIÓN DE TRAIN/TEST
# ========================================================================

def prepare_train_test(agg, target_col='CR2_median'):
    feature_cols = (
        ['n_pisos_first', 'T1_real_first', 'Mf_KN_first', 'N_Dampers_first',
         'densidad_dampers', 'posicion_promedio_dampers', 'posicion_normalizada',
         'concentracion_dampers'] +
        [c for c in agg.columns if c.startswith('damper_piso')]
    )
    groups = agg['estructura_id'].values
    X = agg[feature_cols].values
    y = agg[target_col].values
    return X, y, groups, feature_cols, target_col


# ========================================================================
# 5. ENTRENAMIENTO DE MODELOS
# ========================================================================

def train_models(X, y, groups, feature_cols, target_col='CR2_median'):
    label = TARGETS.get(target_col, target_col)
    print(f"\n[6/8] Entrenamiento — target: {target_col} ({label})")

    results = {}
    gkf = GroupKFold(n_splits=5)

    def _cv_stats(estimator, X, y, groups, gkf):
        cv = cross_validate(estimator, X, y, groups=groups, cv=gkf,
                            scoring={'neg_mse': 'neg_mean_squared_error', 'r2': 'r2'},
                            n_jobs=-1)
        rmse_folds = np.sqrt(-cv['test_neg_mse'])
        return (rmse_folds.mean(), rmse_folds.std(),
                cv['test_r2'].mean(), cv['test_r2'].std())

    print("      [1/4] Linear Regression...")
    lr = LinearRegression()
    rmse_cv, rmse_std, r2_cv, r2_std = _cv_stats(lr, X, y, groups, gkf)
    lr.fit(X, y)
    results['LinearRegression'] = {
        'RMSE_CV': rmse_cv, 'RMSE_CV_std': rmse_std,
        'R2_CV': r2_cv, 'R2_CV_std': r2_std, 'model': lr,
    }
    print(f"            RMSE_CV={rmse_cv:.4f}±{rmse_std:.4f}  R²_CV={r2_cv:.4f}±{r2_std:.4f}")

    print("      [2/4] Ridge (alpha=1.0)...")
    ridge = Ridge(alpha=1.0)
    rmse_cv, rmse_std, r2_cv, r2_std = _cv_stats(ridge, X, y, groups, gkf)
    ridge.fit(X, y)
    results['Ridge'] = {
        'RMSE_CV': rmse_cv, 'RMSE_CV_std': rmse_std,
        'R2_CV': r2_cv, 'R2_CV_std': r2_std, 'model': ridge,
    }
    print(f"            RMSE_CV={rmse_cv:.4f}±{rmse_std:.4f}  R²_CV={r2_cv:.4f}±{r2_std:.4f}")

    print("      [3/4] Random Forest (n=100)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               min_samples_split=20, random_state=42, n_jobs=-1)
    rmse_cv, rmse_std, r2_cv, r2_std = _cv_stats(rf, X, y, groups, gkf)
    rf.fit(X, y)
    results['RandomForest'] = {
        'RMSE_CV': rmse_cv, 'RMSE_CV_std': rmse_std,
        'R2_CV': r2_cv, 'R2_CV_std': r2_std, 'model': rf,
        'feature_importances': rf.feature_importances_,
    }
    print(f"            RMSE_CV={rmse_cv:.4f}±{rmse_std:.4f}  R²_CV={r2_cv:.4f}±{r2_std:.4f}")

    try:
        import xgboost as xgb
        print("      [4/4] XGBoost (n=100)...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6,
                                     learning_rate=0.1, random_state=42,
                                     n_jobs=-1, verbosity=0)
        rmse_cv, rmse_std, r2_cv, r2_std = _cv_stats(xgb_model, X, y, groups, gkf)
        xgb_model.fit(X, y)
        results['XGBoost'] = {
            'RMSE_CV': rmse_cv, 'RMSE_CV_std': rmse_std,
            'R2_CV': r2_cv, 'R2_CV_std': r2_std, 'model': xgb_model,
            'feature_importances': xgb_model.feature_importances_,
        }
        print(f"            RMSE_CV={rmse_cv:.4f}±{rmse_std:.4f}  R²_CV={r2_cv:.4f}±{r2_std:.4f}")
    except ImportError:
        print("      [4/4] XGBoost no instalado. Saltando.")

    # Guardar modelos con sufijo de target
    suffix = target_col.replace('CR2_', '')
    for name, res in results.items():
        path = MODELS_DIR / f'{name}_{suffix}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(res['model'], f)

    return results


# ========================================================================
# 6. EVALUACIÓN Y COMPARACIÓN
# ========================================================================

def evaluate_models(results, feature_cols, target_col='CR2_median'):
    label = TARGETS.get(target_col, target_col)
    suffix = target_col.replace('CR2_', '')
    print(f"\n[7/8] Evaluación — {target_col}")

    comparison = [
        {
            'Model': n,
            'RMSE_CV': r['RMSE_CV'], 'RMSE_CV_std': r['RMSE_CV_std'],
            'R2_CV':   r['R2_CV'],   'R2_CV_std':   r['R2_CV_std'],
        }
        for n, r in results.items()
    ]
    df_comp = pd.DataFrame(comparison).sort_values('RMSE_CV')
    print(df_comp.to_string(index=False))
    df_comp.to_csv(RESULTS_DIR / f'model_comparison_{suffix}.csv', index=False)

    # Feature importance del mejor ensemble
    for model_name in ['XGBoost', 'RandomForest']:
        if model_name in results and 'feature_importances' in results[model_name]:
            importances = results[model_name]['feature_importances']
            indices = np.argsort(importances)[::-1][:15]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importances[indices],
                    color=TARGET_COLORS.get(target_col, 'steelblue'))
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_cols[i] for i in indices])
            ax.set_xlabel('Importancia')
            ax.set_title(f'Top 15 Features — {model_name} | Target: {target_col}\n({label})')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f'feature_importance_{suffix}.png', dpi=120,
                        bbox_inches='tight')
            plt.close()
            break

    return df_comp


# ========================================================================
# 7. ANÁLISIS DE CONVERGENCIA
# ========================================================================

def compare_recommendations(agg, all_models, feature_cols, n_estructuras=20):
    """
    Para n_estructuras aleatorias, verifica si optimizar p10, mediana y p90
    lleva al mismo diseño óptimo (top-1 configuración).
    """
    print(f"\n{'='*70}")
    print("ANÁLISIS DE CONVERGENCIA: ¿Los 3 targets recomiendan el mismo diseño?")
    print(f"{'='*70}")

    sample_ids = (agg['estructura_id'].drop_duplicates()
                  .sample(n=min(n_estructuras, agg['estructura_id'].nunique()),
                          random_state=42).values)

    convergencia = []
    for eid in sample_ids:
        struct_data = agg[agg['estructura_id'] == eid].copy()
        X_s = struct_data[feature_cols].values
        n_pisos = int(struct_data['n_pisos_first'].iloc[0])
        T1 = float(struct_data['T1_real_first'].iloc[0])

        top1_per_target = {}
        top1_ndampers = {}
        for target_col, models in all_models.items():
            best_name = next(
                (m for m in ['XGBoost', 'RandomForest', 'Ridge'] if m in models), None
            )
            if best_name is None:
                continue
            y_pred = models[best_name].predict(X_s)
            struct_data[f'pred_{target_col}'] = y_pred
            best_row = struct_data.nsmallest(1, f'pred_{target_col}').iloc[0]
            top1_per_target[target_col] = best_row['Bits_first']
            top1_ndampers[target_col] = int(best_row['N_Dampers_first'])

        # Convergencia: mismo patrón de bits en los 3 targets
        todos_iguales = len(set(top1_per_target.values())) == 1

        convergencia.append({
            'estructura_id': eid,
            'n_pisos': n_pisos,
            'T1': round(T1, 2),
            'top1_median': top1_per_target.get('CR2_median', '-'),
            'top1_p10':    top1_per_target.get('CR2_p10', '-'),
            'top1_p90':    top1_per_target.get('CR2_p90', '-'),
            'Nd_median':   top1_ndampers.get('CR2_median', -1),
            'Nd_p10':      top1_ndampers.get('CR2_p10', -1),
            'Nd_p90':      top1_ndampers.get('CR2_p90', -1),
            'converge': todos_iguales,
        })

    df_conv = pd.DataFrame(convergencia)
    tasa = df_conv['converge'].mean() * 100

    print(f"\n  Estructuras analizadas : {len(df_conv)}")
    print(f"  Tasa de convergencia   : {tasa:.1f}%  (top-1 idéntico en los 3 targets)")
    print(f"\n  N_Dampers del top-1 por target:")
    for tc in ['CR2_median', 'CR2_p10', 'CR2_p90']:
        col = 'Nd_' + tc.replace('CR2_', '')
        if col in df_conv.columns:
            print(f"    {tc:12s}  media={df_conv[col].mean():.1f}  "
                  f"min={df_conv[col].min()}  max={df_conv[col].max()}")

    # Gráfico de convergencia: N_Dampers recomendado por target
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_conv))
    w = 0.25
    for i, (tc, label) in enumerate(TARGETS.items()):
        col = 'Nd_' + tc.replace('CR2_', '')
        if col in df_conv.columns:
            ax.bar(x + i * w, df_conv[col], width=w,
                   label=tc, color=TARGET_COLORS[tc], alpha=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"E{int(eid)}" for eid in df_conv['estructura_id']],
                       rotation=45, fontsize=7)
    ax.set_ylabel('N_Dampers recomendados (top-1)')
    ax.set_title(f'Convergencia de diseño — {tasa:.0f}% de estructuras convergen')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'convergencia_ndampers.png', dpi=120, bbox_inches='tight')
    plt.close()

    df_conv.to_csv(RESULTS_DIR / 'convergencia_targets.csv', index=False)
    print(f"\n  [OK] Guardado en: {RESULTS_DIR / 'convergencia_targets.csv'}")
    print(f"  [OK] Gráfico en:  {RESULTS_DIR / 'convergencia_ndampers.png'}")

    return df_conv


# ========================================================================
# MAIN
# ========================================================================

def main():
    t0 = time.time()

    df, agg = load_and_aggregate()
    best_per_structure = exploratory_analysis(df, agg)
    agg = feature_engineering(agg)

    all_results = {}
    all_models = {}
    feature_cols = None

    for target_col in TARGETS:
        print(f"\n{'='*70}")
        print(f"  TARGET: {target_col}  —  {TARGETS[target_col]}")
        print(f"{'='*70}")

        X, y, groups, feature_cols, _ = prepare_train_test(agg, target_col=target_col)
        print(f"\n[5/8] Datos listos — features={len(feature_cols)}  samples={len(X):,}")

        results = train_models(X, y, groups, feature_cols, target_col=target_col)
        df_comp = evaluate_models(results, feature_cols, target_col=target_col)

        all_results[target_col] = results
        all_models[target_col] = {name: res['model'] for name, res in results.items()}

    # Resumen comparativo de RMSE_CV entre targets
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO DE RMSE_CV (GroupKFold 5-fold)")
    print(f"{'='*70}")
    rows = []
    for tc, results in all_results.items():
        for name, res in results.items():
            rows.append({
                'Target': tc, 'Model': name,
                'RMSE_CV': res['RMSE_CV'], 'RMSE_CV_std': res['RMSE_CV_std'],
                'R2_CV':   res['R2_CV'],   'R2_CV_std':   res['R2_CV_std'],
            })
    df_summary = pd.DataFrame(rows)
    pivot_rmse = df_summary.pivot(index='Model', columns='Target',
                                  values='RMSE_CV').round(4)
    print(pivot_rmse.to_string())
    df_summary.to_csv(RESULTS_DIR / 'rmse_cv_resumen.csv', index=False)
    print(f"\n  [OK] Métricas completas (RMSE±std, R²±std) en: "
          f"{RESULTS_DIR / 'rmse_cv_resumen.csv'}")

    # Análisis de convergencia
    if feature_cols is not None:
        compare_recommendations(agg, all_models, feature_cols, n_estructuras=20)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETADO EN {elapsed:.1f} SEGUNDOS")
    print(f"{'='*70}")
    print(f"\nSalidas generadas:")
    print(f"  - EDA plots:          {EDA_DIR}/")
    print(f"  - Modelos entrenados: {MODELS_DIR}/  (sufijos: _median / _p10 / _p90)")
    print(f"  - Resultados:         {RESULTS_DIR}/")
    print(f"  - Dataset ML:         {OUT_DIR / 'dataset_ml.parquet'}")
    print(f"  - Convergencia:       {RESULTS_DIR / 'convergencia_targets.csv'}")
    print(f"\nPróximo paso:")
    print(f"  jupyter notebook 04_interactive_analysis.ipynb")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
