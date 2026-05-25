# Pipeline ML — Marcos con amortiguadores de fricción rotacional

Este proyecto entrena un modelo de Machine Learning para predecir el desempeño
sísmico de marcos de acero acoplados a amortiguadores de fricción rotacionales,
y recomendar configuraciones óptimas (ubicación + cantidad mínima + Mf).

## Estructura de carpetas

```
INVESTIGACION FRANCIA/
├── 01_design_of_experiments.py   # Fase 1: genera plan de simulaciones
├── inspect_plan.py               # Inspector del plan (estadísticas + plots)
├── validate_motor.py             # Validación del motor antes de Fase 2
├── 02_run_simulations.py         # Fase 2: simulación masiva con NSD-LCP-PSOR
├── README.md                     # Este archivo
├── sismos/                       # 8 acelerogramas (CSV con t, ag_g)
│   ├── ElCentro 1940.csv
│   ├── Jamiltepec 2018.csv
│   ├── Kobe 1995.csv
│   ├── Loma Prieta 1989.csv
│   ├── Niltepec 2017.csv
│   ├── Northridge 1994.csv
│   ├── Trinidad 1983.csv
│   └── Zacatula 1997.csv
├── out_fase1/                    # SE GENERA: plan de experimentos
│   ├── plan_estructuras.parquet
│   ├── plan_estructuras_full.parquet
│   ├── plan_sismos.parquet
│   ├── plan_experimentos.parquet
│   └── figs/                     # plots de cobertura LHS
├── out_inspect/                  # SE GENERA: plots adicionales del plan
└── out_fase2/                    # SE GENERA: resultados de simulación
    ├── struct_00000.parquet
    ├── struct_00001.parquet
    ├── ...
    └── all_results.parquet       # consolidado final
```

## Requisitos

```bash
pip install numpy pandas scipy pyarrow numba tqdm matplotlib
```

## Flujo completo

### Paso 1: Generar plan de experimentos
```bash
python 01_design_of_experiments.py
```
Genera 1000 estructuras únicas con LHS y 433,984 simulaciones planeadas.
Tiempo: ~30 segundos. Salida: `out_fase1/`.

**Verificar visualmente** los plots en `out_fase1/figs/`:
- 01_cobertura_marginales.png : T1, masa, n_pisos uniformes
- 02_cobertura_conjunta.png   : sin huecos en T1×masa
- 04_calibracion_T1.png       : todos los puntos sobre la diagonal
- 06_ejemplo_perfiles.png     : perfiles de masa y rigidez como esperabas

### Paso 2: Inspeccionar el plan (opcional)
```bash
python inspect_plan.py
```
Muestra estadísticas detalladas y verificaciones de consistencia.
Tiempo: ~10 segundos. Salida: `out_inspect/`.

### Paso 3: Validar el motor antes de la Fase 2 grande
```bash
python validate_motor.py
```
Esto:
1. Verifica que tus 8 sismos cargan bien.
2. Compila Numba (puede tardar 30-60 s la primera vez).
3. Hace un barrido de Mf y verifica que R1, R2, R3 varían (motor correcto).
4. Estima cuánto tardará la Fase 2 en TU PC.

**Si este paso falla, NO continúes a la Fase 2.**

### Paso 4: Lanzar Fase 2 (simulación masiva)
```bash
# Opción A: secuencial (más simple)
python 02_run_simulations.py

# Opción B: paralelo en 4 terminales (más rápido)
python 02_run_simulations.py --shard 0/4
python 02_run_simulations.py --shard 1/4
python 02_run_simulations.py --shard 2/4
python 02_run_simulations.py --shard 3/4
```

**Checkpointing automático**: cada estructura guarda su parquet propio. Si
interrumpes (Ctrl+C) y vuelves a correr, retoma desde donde quedó.

Tiempo estimado: **1-3 horas** en una PC moderna con Numba.

### Paso 5: Consolidar resultados
```bash
python 02_run_simulations.py --consolidate
```
Une todos los `struct_*.parquet` en un único `all_results.parquet`
(~100-200 MB) que será el dataset de entrada para la Fase 4 (ML).

## Configuración del experimento

Parámetros decididos:
- **1000 estructuras** muestreadas con Latin Hypercube Sampling (LHS)
- **8 alturas**: 3, 5, 7, 10, 12, 15, 18, 20 pisos
- **T1 ∈ [0.3, 3.0] s** (calibrado por escala de rigidez)
- **Masa ∈ [150, 350] ton/piso**
- **3 perfiles de masa**: constante, azotea_reducida, lineal_decreciente
- **4 perfiles de rigidez**: constante, lineal_decreciente, logaritmico, bilineal
- **~50 configuraciones de amortiguadores** por estructura (LHS + estructuradas)
- **Mf ∈ [25, 500] KN**
- **8 sismos**

Solver:
- Motor **NSD-LCP-PSOR** (de la tesis original)
- **Rayleigh damping** modos 1 y 3, ζ = 0.03
- **h_a = 0.40 m** (brazo de palanca constante)
- PSOR: max_iter=20, tol=1e-6

Métricas reportadas por simulación:
- `R1, R2, R3` : ratios desplazamiento, aceleración, cortante basal vs caso base
- `CR1, CR2`  : índices globales (CR2 ∈ [0, √3])
- `drift_max, drift_ratio` : distorsión máxima de entrepiso
- `peak_u_global` : desplazamiento máximo en cualquier piso
- `E_total_J` : energía disipada (estimación post-proceso)

## Próximos pasos (Fase 4: ML)

Cuando termines Fase 2, el archivo `out_fase2/all_results.parquet` será la
entrada para entrenar el modelo de ML:
- Modelo A: predictor de CR2 dado (estructura, configuración, sismo)
- Optimizador: encuentra configuración óptima minimizando N_Dampers
  sujeto a CR2 ≤ umbral
