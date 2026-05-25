"""
OPTIMIZADOR AVANZADO — Pareto Front + Minimización de N_Dampers

Agrega este código al notebook en una celda nueva después de cargar los modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pareto_front(df_configs, obj1_col='CR2_pred', obj2_col='N_Dampers_first', 
                 minimize_both=True):
    """
    Encuentra el Pareto front de un conjunto de configuraciones.
    
    Pareto dominance: config A domina a config B si:
      - A es mejor o igual en TODOS los objetivos
      - A es estrictamente mejor en AL MENOS UN objetivo
    
    Args:
        df_configs: DataFrame con las configuraciones
        obj1_col: columna del primer objetivo (ej: CR2_pred)
        obj2_col: columna del segundo objetivo (ej: N_Dampers_first)
        minimize_both: si True, ambos se minimizan; si False, obj1 se minimiza, obj2 se maximiza
    
    
    Returns:
        DataFrame solo con configs no-dominadas (Pareto front)
    """
    configs = df_configs.copy()
    is_pareto = np.ones(len(configs), dtype=bool)
    
    for i in range(len(configs)):
        if not is_pareto[i]:
            continue
        
        # Comparar config i con todas las demás
        for j in range(len(configs)):
            if i == j or not is_pareto[j]:
                continue
            
            # Config j domina a config i si:
            if minimize_both:
                # j es mejor o igual en ambos
                better_or_equal = (configs.iloc[j][obj1_col] <= configs.iloc[i][obj1_col] and
                                  configs.iloc[j][obj2_col] <= configs.iloc[i][obj2_col])
                # j es estrictamente mejor en al menos uno
                strictly_better = (configs.iloc[j][obj1_col] < configs.iloc[i][obj1_col] or
                                  configs.iloc[j][obj2_col] < configs.iloc[i][obj2_col])
            else:
                # obj1 minimizar, obj2 maximizar
                better_or_equal = (configs.iloc[j][obj1_col] <= configs.iloc[i][obj1_col] and
                                  configs.iloc[j][obj2_col] >= configs.iloc[i][obj2_col])
                strictly_better = (configs.iloc[j][obj1_col] < configs.iloc[i][obj1_col] or
                                  configs.iloc[j][obj2_col] > configs.iloc[i][obj2_col])
            
            if better_or_equal and strictly_better:
                is_pareto[i] = False
                break
    
    return configs[is_pareto].sort_values(obj2_col)


def recomendar_pareto(estructura_id, best_model, feature_cols, df, 
                      max_configs=None, cr2_threshold=None):
    """
    Recomienda configuraciones usando Pareto front.
    
    Args:
        estructura_id: ID de la estructura
        best_model: modelo entrenado
        feature_cols: lista de nombres de features
        df: dataset completo
        max_configs: número máximo de configs a retornar del Pareto front
        cr2_threshold: si se da, solo retorna configs con CR2_pred <= threshold
    
    Returns:
        DataFrame con configs del Pareto front
    """
    # Filtrar estructura
    struct_data = df[df['estructura_id'] == estructura_id].copy()
    
    if len(struct_data) == 0:
        print(f"Estructura {estructura_id} no encontrada.")
        return None
    
    # Info
    n_pisos = struct_data['n_pisos_first'].iloc[0]
    T1 = struct_data['T1_real_first'].iloc[0]
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZADOR PARETO — Estructura {estructura_id}")
    print(f"{'='*70}")
    print(f"  n_pisos: {n_pisos}")
    print(f"  T1: {T1:.3f} s")
    print(f"  Configuraciones disponibles: {len(struct_data)}")
    
    # Predecir
    X_struct = struct_data[feature_cols].values
    y_pred = best_model.predict(X_struct)
    struct_data['CR2_pred'] = y_pred
    
    # Aplicar threshold si se dio
    if cr2_threshold is not None:
        struct_data = struct_data[struct_data['CR2_pred'] <= cr2_threshold]
        print(f"  Configs con CR2_pred <= {cr2_threshold}: {len(struct_data)}")
    
    if len(struct_data) == 0:
        print("  [WARN] Ninguna config cumple el threshold. Relaja el threshold.")
        return None
    
    # Calcular Pareto front
    pareto_configs = pareto_front(struct_data, 'CR2_pred', 'N_Dampers_first', 
                                   minimize_both=True)
    
    print(f"  Configs en Pareto front: {len(pareto_configs)}")
    print(f"{'='*70}\n")
    
    # Limitar cantidad si se pidió
    if max_configs is not None and len(pareto_configs) > max_configs:
        pareto_configs = pareto_configs.head(max_configs)
    
    # Mostrar tabla
    tabla = pareto_configs[['Bits_first', 'N_Dampers_first', 'Mf_KN_first', 
                            'CR2_median', 'CR2_pred']].copy()
    tabla['Error_pred'] = np.abs(tabla['CR2_median'] - tabla['CR2_pred'])
    
    print("CONFIGURACIONES PARETO-ÓPTIMAS (ordenadas por N_Dampers):")
    print(tabla.to_string(index=False))
    
    # Plot del Pareto front
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Todas las configs (fondo gris)
    ax.scatter(struct_data['N_Dampers_first'], struct_data['CR2_pred'],
               alpha=0.3, s=30, color='gray', label='Todas las configs')
    
    # Pareto front (rojo, destacado)
    ax.scatter(pareto_configs['N_Dampers_first'], pareto_configs['CR2_pred'],
               color='red', s=120, marker='*', label='Pareto front', zorder=10,
               edgecolors='black', linewidths=1.5)
    
    # Línea conectando Pareto
    pareto_sorted = pareto_configs.sort_values('N_Dampers_first')
    ax.plot(pareto_sorted['N_Dampers_first'], pareto_sorted['CR2_pred'],
            'r--', alpha=0.5, linewidth=2, zorder=5)
    
    # Anotar puntos extremos del Pareto
    min_dampers = pareto_configs.nsmallest(1, 'N_Dampers_first').iloc[0]
    min_cr2 = pareto_configs.nsmallest(1, 'CR2_pred').iloc[0]
    
    ax.annotate(f'Mín dampers: {int(min_dampers["N_Dampers_first"])}\\nCR2={min_dampers["CR2_pred"]:.3f}',
                xy=(min_dampers['N_Dampers_first'], min_dampers['CR2_pred']),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.annotate(f'Mín CR2: {min_cr2["CR2_pred"]:.3f}\\nDampers={int(min_cr2["N_Dampers_first"])}',
                xy=(min_cr2['N_Dampers_first'], min_cr2['CR2_pred']),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('N_Dampers (costo)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CR2 predicho (desempeño)', fontsize=12, fontweight='bold')
    ax.set_title(f'Pareto Front — Estructura {estructura_id} ({n_pisos} pisos, T1={T1:.2f}s)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()  # Menor CR2 arriba (mejor)
    
    plt.tight_layout()
    plt.show()
    
    return pareto_configs


def recomendar_minimo_dampers(estructura_id, best_model, feature_cols, df, 
                               cr2_target=1.0):
    """
    Encuentra la config con MÍNIMO N_Dampers que cumpla CR2 <= cr2_target.
    
    Args:
        estructura_id: ID de la estructura
        best_model: modelo entrenado
        feature_cols: lista de features
        df: dataset
        cr2_target: threshold de CR2 aceptable (ej: 1.0 = sin mejora, 0.8 = 20% mejora)
    
    Returns:
        DataFrame con la mejor config (si existe)
    """
    struct_data = df[df['estructura_id'] == estructura_id].copy()
    
    if len(struct_data) == 0:
        print(f"Estructura {estructura_id} no encontrada.")
        return None
    
    n_pisos = struct_data['n_pisos_first'].iloc[0]
    T1 = struct_data['T1_real_first'].iloc[0]
    
    # Predecir
    X_struct = struct_data[feature_cols].values
    y_pred = best_model.predict(X_struct)
    struct_data['CR2_pred'] = y_pred
    
    # Filtrar por threshold
    feasible = struct_data[struct_data['CR2_pred'] <= cr2_target].copy()
    
    print(f"\n{'='*70}")
    print(f"MINIMIZACIÓN DE AMORTIGUADORES — Estructura {estructura_id}")
    print(f"{'='*70}")
    print(f"  n_pisos: {n_pisos}, T1: {T1:.3f} s")
    print(f"  CR2 target: <= {cr2_target}")
    print(f"  Configs que cumplen: {len(feasible)} / {len(struct_data)}")
    
    if len(feasible) == 0:
        print(f"\n  [WARN] Ninguna config alcanza CR2 <= {cr2_target}")
        print(f"  Mejor CR2 alcanzable: {struct_data['CR2_pred'].min():.3f}")
        print(f"  Sugerencia: relaja cr2_target a {struct_data['CR2_pred'].min() + 0.05:.2f}")
        return None
    
    # Ordenar por N_Dampers (ascendente) y tomar la de menos amortiguadores
    best = feasible.nsmallest(1, 'N_Dampers_first')
    
    print(f"\n  MEJOR CONFIGURACIÓN (mínimo N_Dampers que cumple CR2 <= {cr2_target}):")
    print(f"  {'='*66}")
    print(f"  Bits:       {best['Bits_first'].iloc[0]}")
    print(f"  N_Dampers:  {int(best['N_Dampers_first'].iloc[0])}")
    print(f"  Mf [kN]:    {best['Mf_KN_first'].iloc[0]:.1f}")
    print(f"  CR2_pred:   {best['CR2_pred'].iloc[0]:.3f}")
    print(f"  CR2_real:   {best['CR2_median'].iloc[0]:.3f}")
    print(f"  {'='*66}\n")
    
    return best


# =============================================================================
# EJEMPLO DE USO (copiar al notebook)
# =============================================================================

"""
# CELDA 1: Cargar modelo y dataset (ya lo tienes)
import pickle
from pathlib import Path
import pandas as pd

best_model = pickle.load(open('out_ml/models/RandomForest.pkl', 'rb'))
df = pd.read_parquet('out_ml/dataset_ml.parquet')

feature_cols = (
    ['n_pisos_first', 'T1_real_first', 'Mf_KN_first', 'N_Dampers_first',
     'densidad_dampers', 'posicion_promedio_dampers', 'posicion_normalizada',
     'concentracion_dampers', 'energia_total_normalizada'] +
    [c for c in pytdf.columns if c.startswith('damper_piso')]
)

# CELDA 2: Pareto front (ver trade-off completo)
pareto_configs = recomendar_pareto(
    estructura_id=250,
    best_model=best_model,
    feature_cols=feature_cols,
    df=df,
    max_configs=10,  # Mostrar top 10 del Pareto
    cr2_threshold=1.5  # Opcional: solo configs con CR2 <= 1.5
)

# CELDA 3: Minimizar amortiguadores con constraint de desempeño
mejor_config = recomendar_minimo_dampers(
    estructura_id=250,
    best_model=best_model,
    feature_cols=feature_cols,
    df=df,
    cr2_target=0.9  # Quiero CR2 <= 0.9 (10% mejor que caso base) con mínimo N_Dampers
)
"""
