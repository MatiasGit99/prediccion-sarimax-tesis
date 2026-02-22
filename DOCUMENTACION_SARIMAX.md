# Documentación Técnica: Predicción del Precio del Cemento con SARIMAX

**Autor:** Tesis de Grado / Posgrado
**Entorno de ejecución:** Kaggle (CPU/GPU)
**Archivo principal:** `prediccion_cemento_sarimax.py`
**Fecha:** 2026

---

## Tabla de Contenidos

1. [Introducción y Motivación](#1-introducción-y-motivación)
2. [Fundamentos Teóricos](#2-fundamentos-teóricos)
   - 2.1 [Series Temporales](#21-series-temporales)
   - 2.2 [Modelo ARIMA](#22-modelo-arima)
   - 2.3 [Modelo SARIMA (Estacional)](#23-modelo-sarima-estacional)
   - 2.4 [Modelo SARIMAX (con Variables Exógenas)](#24-modelo-sarimax-con-variables-exógenas)
   - 2.5 [Criterios de Información AIC y BIC](#25-criterios-de-información-aic-y-bic)
3. [Arquitectura del Pipeline](#3-arquitectura-del-pipeline)
4. [Descripción Detallada de Cada Sección del Código](#4-descripción-detallada-de-cada-sección-del-código)
   - 4.1 [Sección 0: Instalación de Dependencias](#41-sección-0-instalación-de-dependencias)
   - 4.2 [Sección 1: Importaciones](#42-sección-1-importaciones)
   - 4.3 [Sección 2: Configuración Global](#43-sección-2-configuración-global)
   - 4.4 [Sección 3: Verificación de Hardware (GPU)](#44-sección-3-verificación-de-hardware-gpu)
   - 4.5 [Sección 4: Carga y Preparación de Datos](#45-sección-4-carga-y-preparación-de-datos)
   - 4.6 [Sección 5: Análisis Exploratorio (EDA)](#46-sección-5-análisis-exploratorio-eda)
   - 4.7 [Sección 6: Modelado SARIMAX](#47-sección-6-modelado-sarimax)
   - 4.8 [Sección 7: Métricas de Evaluación](#48-sección-7-métricas-de-evaluación)
   - 4.9 [Sección 8: Validación Cruzada Rolling](#49-sección-8-validación-cruzada-rolling)
   - 4.10 [Sección 9: Gráficos de Resultados](#410-sección-9-gráficos-de-resultados)
   - 4.11 [Sección 10: Exportación de Resultados](#411-sección-10-exportación-de-resultados)
   - 4.12 [Sección 11: Pipeline Principal (main)](#412-sección-11-pipeline-principal-main)
5. [Descripción de los Gráficos Generados](#5-descripción-de-los-gráficos-generados)
6. [Descripción de los Archivos de Salida](#6-descripción-de-los-archivos-de-salida)
7. [Métricas de Evaluación: Interpretación Profunda](#7-métricas-de-evaluación-interpretación-profunda)
8. [Tests Estadísticos: Interpretación](#8-tests-estadísticos-interpretación)
9. [Rol de la GPU en el Pipeline](#9-rol-de-la-gpu-en-el-pipeline)
10. [Configuración de Rutas en Kaggle](#10-configuración-de-rutas-en-kaggle)
11. [Guía de Interpretación para la Tesis](#11-guía-de-interpretación-para-la-tesis)
12. [Referencias Bibliográficas](#12-referencias-bibliográficas)

---

## 1. Introducción y Motivación

El precio del cemento es un indicador económico sensible a múltiples factores: fluctuaciones en la demanda de construcción, costos logísticos (transporte fluvial), estacionalidad climática y eventos extraordinarios como la pandemia de COVID-19. La capacidad de predecir su evolución a mediano plazo permite a empresas del sector anticipar decisiones de compra, planificación de inventario y estrategias de precios.

### ¿Por qué SARIMAX?

El modelo **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) es la elección natural para este problema porque:

| Característica del problema | Componente SARIMAX que la modela |
|---|---|
| Tendencia de largo plazo en los precios | Diferenciación `d` y componente AR |
| Patrón estacional recurrente (ej. mayor actividad en verano/seco) | Parte estacional `(P,D,Q,m)` |
| Impacto del nivel del río en el costo logístico fluvial | Variables exógenas `X_t` (Nivel_Rio) |
| Efecto de las restricciones COVID-19 | Variable exógena binaria `Cuarentena` |
| Autocorrelación de los errores de predicción | Parte MA `q` |

---

## 2. Fundamentos Teóricos

### 2.1 Series Temporales

Una **serie temporal** es una sucesión de observaciones $\{Y_t\}_{t=1}^{T}$ tomadas a intervalos regulares de tiempo. En este estudio, $Y_t$ representa el precio promedio del cemento en el mes $t$.

Las propiedades fundamentales que determinan la metodología de modelado son:

**Estacionariedad:** Una serie es (débilmente) estacionaria si:
- Su media $\mu = E[Y_t]$ es constante en el tiempo.
- Su varianza $\sigma^2 = \text{Var}(Y_t)$ es finita y constante.
- Su autocovarianza $\gamma(k) = \text{Cov}(Y_t, Y_{t-k})$ depende solo del rezago $k$, no de $t$.

Si la serie no es estacionaria (lo usual en precios), se aplica **diferenciación**:

$$\nabla Y_t = Y_t - Y_{t-1} = (1-B)Y_t$$

donde $B$ es el operador de retardo (*backshift operator*): $BY_t = Y_{t-1}$.

### 2.2 Modelo ARIMA

El modelo **ARIMA(p, d, q)** combina tres componentes:

**Autorregresivo (AR(p)):** El valor actual depende de $p$ valores pasados.
$$Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \ldots + \phi_p Y_{t-p} + \varepsilon_t$$

**Integrado (I(d)):** Aplica $d$ diferenciaciones para lograr estacionariedad.
$$\nabla^d Y_t = (1-B)^d Y_t$$

**Media Móvil (MA(q)):** El valor actual depende de $q$ errores pasados.
$$Y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \ldots + \theta_q \varepsilon_{t-q}$$

**Formulación unificada ARIMA(p,d,q):**

$$\phi(B)(1-B)^d Y_t = \theta(B)\varepsilon_t$$

donde:
- $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \ldots - \phi_p B^p$ (polinomio AR)
- $\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \ldots + \theta_q B^q$ (polinomio MA)
- $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ (ruido blanco gaussiano)

### 2.3 Modelo SARIMA (Estacional)

El **SARIMA(p,d,q)(P,D,Q)_m** extiende el ARIMA para capturar patrones estacionales de período $m$ (en este caso $m=12$ para datos mensuales):

$$\Phi(B^m)\phi(B)(1-B^m)^D(1-B)^d Y_t = \Theta(B^m)\theta(B)\varepsilon_t$$

donde los nuevos operadores son:
- $\Phi(B^m) = 1 - \Phi_1 B^m - \ldots - \Phi_P B^{Pm}$ (AR estacional de orden $P$)
- $\Theta(B^m) = 1 + \Theta_1 B^m + \ldots + \Theta_Q B^{Qm}$ (MA estacional de orden $Q$)
- $(1-B^m)^D$ aplica $D$ diferenciaciones estacionales

**Interpretación de los órdenes:**

| Parámetro | Significado | Impacto |
|---|---|---|
| `p` | Rezagos AR ordinarios | Memoria de corto plazo |
| `d` | Diferenciaciones ordinarias | Elimina tendencia |
| `q` | Rezagos MA ordinarios | Corrección de errores recientes |
| `P` | Rezagos AR estacionales | Dependencia de mismo mes del año anterior |
| `D` | Diferenciaciones estacionales | Elimina estacionalidad determinista |
| `Q` | Rezagos MA estacionales | Corrección de errores estacionales |
| `m` | Período estacional | 12 para datos mensuales |

### 2.4 Modelo SARIMAX (con Variables Exógenas)

El **SARIMAX** agrega un vector de regresores externos $\mathbf{X}_t$:

$$\Phi(B^m)\phi(B)(1-B^m)^D(1-B)^d Y_t = \boldsymbol{\beta}^\top \mathbf{X}_t + \Theta(B^m)\theta(B)\varepsilon_t$$

En nuestro caso:
$$\mathbf{X}_t = \begin{pmatrix} \text{Nivel\_Rio}_t \\ \text{Cuarentena}_t \end{pmatrix}, \quad \boldsymbol{\beta} = \begin{pmatrix} \beta_1 \\ \beta_2 \end{pmatrix}$$

**Interpretación de los coeficientes $\boldsymbol{\beta}$:**
- $\hat{\beta}_1$: Cambio esperado en el precio del cemento por cada metro adicional en el nivel del río, manteniendo todo lo demás constante (*ceteris paribus*).
- $\hat{\beta}_2$: Efecto promedio de las restricciones de cuarentena sobre el precio.

La estimación se realiza por **Máxima Verosimilitud (MLE)**, maximizando:

$$\mathcal{L}(\boldsymbol{\phi}, \boldsymbol{\theta}, \boldsymbol{\beta}, \sigma^2 \mid \mathbf{Y}, \mathbf{X}) = -\frac{T}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^T \varepsilon_t^2$$

El optimizador L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) minimiza el negativo de la log-verosimilitud usando información de gradientes de segundo orden aproximados.

### 2.5 Criterios de Información AIC y BIC

Para comparar modelos con diferente número de parámetros:

$$\text{AIC} = -2\ln(\hat{\mathcal{L}}) + 2k$$

$$\text{BIC} = -2\ln(\hat{\mathcal{L}}) + k\ln(T)$$

donde $k$ es el número de parámetros y $T$ el número de observaciones.

- **AIC** tiende a seleccionar modelos más complejos (menor penalización).
- **BIC** penaliza más fuertemente la complejidad (adecuado para $T$ grande).
- En ambos casos, **menor valor = mejor modelo**.
- `auto_arima` usa AIC por defecto en este pipeline.

---

## 3. Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  PIPELINE SARIMAX                       │
│                                                         │
│  ENTRADA                                                │
│  ├── precios_cemento_interpolado.csv  (datos históricos)│
│  └── promedio_mensual_nivel_predicho.csv (exóg. futuras)│
│                                                         │
│  PASO 1: Verificar GPU/CPU                              │
│  PASO 2: Cargar datos históricos   ──→ df_hist          │
│  PASO 3: Cargar exógenas futuras   ──→ df_future_exog   │
│                                                         │
│  PASO 4: EDA ──→ 5 gráficos PNG                         │
│    ├── Tests ADF y KPSS (estacionariedad)               │
│    ├── 01_series_temporales.png                         │
│    ├── 02_descomposicion_estacional.png                 │
│    ├── 03_acf_pacf.png                                  │
│    ├── 04_correlacion.png                               │
│    └── 05_patron_estacional.png                         │
│                                                         │
│  PASO 5: Split train/test (N_TEST=15 meses)             │
│                                                         │
│  PASO 6: auto_arima (búsqueda de parámetros)            │
│    └── → (p,d,q)(P,D,Q,m) óptimos                      │
│                                                         │
│  PASO 7: Modelo validación (train) → métricas (test)    │
│                                                         │
│  PASO 8: Modelo final (todos los datos)                 │
│                                                         │
│  PASO 9: Pronóstico futuro + IC 95%                     │
│                                                         │
│  PASO 10: Rolling Cross-Validation                      │
│                                                         │
│  PASO 11: Gráficos de resultados ──→ 6 gráficos PNG     │
│    ├── 06_pronostico_completo.png                       │
│    ├── 07_diagnostico_residuos.png                      │
│    ├── 08_comparacion_real_predicho.png                 │
│    ├── 09_pronostico_intervalo_confianza.png             │
│    ├── 10_rolling_forecast_cv.png                       │
│    └── 11_ajuste_insample.png                           │
│                                                         │
│  PASO 12: Exportar                                      │
│    ├── pronostico_futuro_cemento.csv                    │
│    ├── prediccion_conjunto_prueba.csv                   │
│    ├── metricas_modelo.csv                              │
│    └── informe_sarimax.txt                              │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Descripción Detallada de Cada Sección del Código

### 4.1 Sección 0: Instalación de Dependencias

```python
def _instalar_si_falta(paquete: str, nombre_import: str | None = None) -> None:
```

**¿Qué hace?**
Intenta importar el paquete. Si no está disponible (`ImportError`), lo instala silenciosamente con `pip`. Esto es una práctica de robustez para entornos Kaggle donde la imagen base puede no incluir `pmdarima`.

**¿Por qué no usar `!pip install` directo?**
En un script `.py` (no notebook), no existe el magic command `!`. La función `subprocess.check_call` invoca `pip` como subproceso del sistema operativo, que es la forma correcta en un script puro.

**Paquetes instalados si faltan:**
- `pmdarima`: Para la búsqueda automática de parámetros con `auto_arima`.
- `openpyxl`: Para exportación opcional a Excel (usado por pandas).

---

### 4.2 Sección 1: Importaciones

Las librerías se organizan por categoría:

| Librería | Categoría | Uso principal |
|---|---|---|
| `numpy` | Cálculo numérico | Operaciones matriciales, métricas |
| `pandas` | Datos tabulares | DataFrames, índices temporales |
| `matplotlib` | Visualización | Creación de todos los gráficos |
| `seaborn` | Visualización | Mapa de calor de correlaciones |
| `statsmodels` | Econometría | SARIMAX, tests ADF/KPSS, ACF/PACF |
| `pmdarima` | AutoML para ARIMA | Búsqueda automática de parámetros |
| `sklearn.metrics` | Evaluación ML | RMSE, MAE, R² |
| `scipy.stats` | Estadística | Distribución normal, Q-Q plot, Shapiro-Wilk |

**Configuración crítica:**
```python
matplotlib.use("Agg")
```
Este backend no requiere pantalla gráfica (Display). Es **obligatorio** en Kaggle, servidores headless y scripts de línea de comandos. Sin él, el script lanzaría un error `_tkinter.TclError`.

---

### 4.3 Sección 2: Configuración Global

#### Rutas de Kaggle

```python
RUTA_ENTRADA   = "/kaggle/input/"
RUTA_SALIDA    = "/kaggle/working/"
NOMBRE_DATASET = "dataset-sarimax-cemento"
```

En Kaggle, los datasets subidos por el usuario se montan automáticamente en `/kaggle/input/<nombre-del-dataset>/`. Los archivos de salida se guardan en `/kaggle/working/` y se pueden descargar desde la interfaz web.

La estructura esperada en Kaggle es:
```
/kaggle/input/dataset-sarimax-cemento/
├── precios_cemento_interpolado.csv
└── promedio_mensual_nivel_predicho.csv

/kaggle/working/
├── graficos/
│   ├── 01_series_temporales.png
│   ├── ...
│   └── 11_ajuste_insample.png
├── pronostico_futuro_cemento.csv
├── prediccion_conjunto_prueba.csv
├── metricas_modelo.csv
└── informe_sarimax.txt
```

#### Parámetros del modelo

```python
PERIODO_ESTACIONAL = 12   # m = 12 meses (frecuencia mensual)
N_TEST             = 15   # Meses reservados para evaluación
ALPHA_IC           = 0.05 # Nivel de significancia (IC = 95%)
```

**¿Por qué `m=12` y no `m=15` como en el original?**
El período estacional debe reflejar la frecuencia del patrón recurrente. Los datos son mensuales y los precios de cemento siguen un ciclo anual de 12 meses (vinculado a temporadas de construcción, lluvias/estiaje del río). El valor `m=15` del notebook original carece de justificación económica y puede llevar a sobreajuste estacional.

#### Paleta de colores accesible

```python
COLORES = {
    "train"   : "#1565C0",  # Azul oscuro (WCAG AA compatible)
    "test"    : "#E65100",  # Naranja oscuro
    ...
}
```

Los colores se eligieron siguiendo las recomendaciones **WCAG 2.1** para accesibilidad visual y compatibilidad con daltonismo (paleta distinguible en modo blanco y negro al imprimir la tesis).

---

### 4.4 Sección 3: Verificación de Hardware (GPU)

```python
def verificar_hardware() -> dict:
```

#### Rol de la GPU en SARIMAX

> **Nota técnica importante para la tesis:**
> El algoritmo SARIMAX implementado en `statsmodels` es **puramente CPU-bound**. La estimación por Máxima Verosimilitud (L-BFGS) opera secuencialmente sobre la log-verosimilitud de la serie temporal y no tiene implementación CUDA ni paralelismo de datos en GPU.

Sin embargo, el acelerador GPU de Kaggle **sí aporta ventajas indirectas**:

1. **Mayor RAM disponible:** La instancia T4 de Kaggle ofrece 16 GB de VRAM + 30 GB de RAM del sistema, vs. 30 GB de RAM solo en modo CPU. Esto permite cargar datasets más grandes sin errores de memoria.

2. **Paralelismo CPU mejorado:** `auto_arima` con `n_jobs=-1` usa todos los núcleos de CPU disponibles. La instancia GPU de Kaggle suele tener más núcleos CPU que la CPU-only.

3. **cuML/RAPIDS:** Si `cupy` está disponible, las operaciones NumPy de preprocesamiento (normalización, cálculo de métricas) pueden ejecutarse en GPU, reduciendo tiempos de computación para datasets muy grandes.

4. **Experimentos complementarios:** Si en el futuro se agregan modelos de Deep Learning (LSTM, N-BEATS) para comparación, la GPU es esencial.

```python
if HAS_GPU:
    try:
        import cupy as cp
        # Las operaciones np.array() se pueden reemplazar por cp.array()
        # para aprovechar la GPU en preprocesamiento
```

---

### 4.5 Sección 4: Carga y Preparación de Datos

#### Función `cargar_datos_historicos`

```python
def cargar_datos_historicos(ruta: str) -> pd.DataFrame:
```

**Transformaciones aplicadas paso a paso:**

**Paso 1 — Lectura del CSV:**
```python
df = pd.read_csv(ruta, index_col="Fecha", parse_dates=True, decimal=",")
```
- `index_col="Fecha"`: La columna `Fecha` se usa como índice.
- `parse_dates=True`: Convierte el índice a `DatetimeIndex` automáticamente.
- `decimal=","`: Interpreta la coma como separador decimal (formato europeo/latinoamericano, ej: `1.234,56` → `1234.56`).

**Paso 2 — Renombrado flexible de columnas:**
```python
rename = {}
for c in df.columns:
    cl = c.lower()
    if "precio" in cl or "polinomial" in cl:
        rename[c] = "Precio_Cemento"
    elif "nivel" in cl:
        rename[c] = "Nivel_Rio"
    elif "cuarentena" in cl or "covid" in cl:
        rename[c] = "Cuarentena"
```
Este mapeo flexible tolera variaciones en los nombres de columna del CSV sin modificar el código.

**Paso 3 — Conversión de tipos numéricos:**
```python
def _limpiar_columna_numerica(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.replace(",", ".", regex=False).astype(float)
```
Convierte `"1234,56"` → `"1234.56"` → `1234.56`. Necesario porque algunos CSV exportados desde Excel en España/Latinoamérica usan coma como separador decimal.

**Paso 4 — Frecuencia temporal:**
```python
df = df.asfreq("MS")
```
`"MS"` = Month Start (inicio de mes). Esto asegura que `pandas` reconoce la serie como mensual y permite operaciones de series temporales (diferenciación, rezagos) correctamente.

**Paso 5 — Interpolación de valores faltantes:**
```python
df = df.interpolate(method="linear").bfill().ffill()
```
- `interpolate("linear")`: Rellena valores intermedios usando interpolación lineal entre el valor anterior y el siguiente. Adecuado para precios con tendencia suave.
- `.bfill()`: Rellena hacia atrás (si hay NaN al inicio de la serie).
- `.ffill()`: Rellena hacia adelante (si hay NaN al final).

#### Función `cargar_exogenas_futuras`

```python
def cargar_exogenas_futuras(ruta: str) -> pd.DataFrame:
```

Carga el archivo `promedio_mensual_nivel_predicho.csv`, que contiene las predicciones del nivel del río para el período de pronóstico. La columna `Cuarentena = 0` refleja el supuesto de que no habrá restricciones sanitarias en el período futuro.

**Supuesto crítico:** Para que el pronóstico de precios sea válido, las predicciones del nivel del río deben provenir de un modelo hidrológico o climatológico separado, y su calidad impacta directamente en la calidad del pronóstico de precios.

---

### 4.6 Sección 5: Análisis Exploratorio (EDA)

#### Función `test_estacionariedad`

```python
def test_estacionariedad(serie: pd.Series, nombre: str = "Serie") -> dict:
```

**Test ADF (Augmented Dickey-Fuller):**

Contrasta la hipótesis nula de raíz unitaria:
- $H_0$: La serie tiene raíz unitaria → NO es estacionaria
- $H_1$: La serie es estacionaria

El estadístico ADF es negativo; cuanto más negativo, más evidencia contra $H_0$.

**Regla de decisión:** Se rechaza $H_0$ si $p < 0.05$ → la serie es estacionaria.

**Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin):**

Contrasta en sentido opuesto al ADF:
- $H_0$: La serie es estacionaria alrededor de una tendencia
- $H_1$: La serie tiene raíz unitaria

**Regla de decisión:** Se rechaza $H_0$ si $p < 0.05$ → la serie NO es estacionaria.

**Tabla de interpretación conjunta:**

| ADF p-valor | KPSS p-valor | Conclusión |
|---|---|---|
| < 0.05 | > 0.05 | Serie estacionaria (ambos tests confirman) |
| > 0.05 | < 0.05 | Serie NO estacionaria (ambos tests confirman) |
| < 0.05 | < 0.05 | Posible tendencia determinista (inspeccionar) |
| > 0.05 | > 0.05 | Resultados inconcluyentes (serie con estructura compleja) |

#### Gráficos EDA (ver Sección 5 de esta documentación)

Los cinco gráficos EDA (`01` al `05`) son fundamentales para:
1. Justificar la elección del período estacional `m=12`.
2. Evidenciar la correlación entre el nivel del río y el precio del cemento.
3. Diagnosticar si se requiere diferenciación antes del modelado.
4. Identificar el impacto visual del período COVID-19.

---

### 4.7 Sección 6: Modelado SARIMAX

#### Función `buscar_parametros_auto`

```python
def buscar_parametros_auto(Y_train, X_train, m=12) -> tuple:
```

**Algoritmo Stepwise de Hyndman-Khandakar:**

Este algoritmo, implementado en `pmdarima.auto_arima`, evita la búsqueda exhaustiva por grilla (que requeriría evaluar $4 \times 4 \times 4 \times 3 \times 2 \times 3 = 1{,}152$ modelos). En su lugar:

1. Comienza con el modelo ARIMA(2,d,2)(1,D,1)[m] como punto de partida.
2. Determina `d` y `D` usando tests de raíz unitaria (ADF y OCSB).
3. Evalúa los 4 modelos vecinos más cercanos (variando `p`, `q`, `P`, `Q` en ±1).
4. Si algún vecino tiene AIC menor, se mueve a ese modelo y repite.
5. Termina cuando ningún vecino mejora el AIC.

**Parámetros de búsqueda usados:**

```python
auto_arima(
    Y_train,
    exogenous = X_train,
    start_p=0, max_p=3,    # Espacio de búsqueda para p
    start_q=0, max_q=3,    # Espacio de búsqueda para q
    d=None,                # Detección automática via test ADF
    start_P=0, max_P=2,    # Espacio de búsqueda para P
    start_Q=0, max_Q=2,    # Espacio de búsqueda para Q
    D=None,                # Detección automática via test OCSB
    seasonal=True,
    m=12,                  # Período estacional mensual
    stepwise=True,         # Algoritmo Hyndman-Khandakar
    information_criterion="aic",
    n_jobs=-1,             # Paralelismo CPU
)
```

**Importante:** `auto_arima` se entrena **solo con datos de TRAIN** (`Y_train`, `X_train`). Esto es fundamental para evitar *data leakage* (fuga de información del futuro al pasado).

#### Función `entrenar_sarimax`

```python
def entrenar_sarimax(Y, X, order, seasonal_order, nombre="Modelo SARIMAX"):
```

**Parámetros de estimación:**

```python
modelo.fit(
    disp=False,      # Silencia output de optimizador
    method="lbfgs",  # Optimizador L-BFGS
    maxiter=300,     # Máximo de iteraciones
)
```

**L-BFGS (Limited-memory BFGS):**
- Es una variante del método cuasi-Newton que usa una aproximación de bajo rango de la matriz Hessiana inversa.
- Requiere solo almacenar los últimos $m=10$ vectores de gradiente en lugar de la Hessiana completa.
- Converge en $O(k^2)$ evaluaciones de función, donde $k$ es el número de parámetros.
- Es particularmente robusto para series largas con muchos parámetros.

**`enforce_stationarity=False` y `enforce_invertibility=False`:**
Permiten que el optimizador explore regiones del espacio paramétrico cerca de (pero no en) la frontera de estacionariedad/invertibilidad. Sin estas flags, el optimizador rechaza directamente soluciones en la frontera, lo que puede impedir encontrar el óptimo global.

---

### 4.8 Sección 7: Métricas de Evaluación

```python
def calcular_metricas(y_real, y_pred, nombre="") -> dict:
```

#### Tabla de métricas con fórmulas e interpretación

| Métrica | Fórmula | Interpretación | Rango |
|---|---|---|---|
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{t=1}^n (Y_t - \hat{Y}_t)^2}$ | Error cuadrático medio. Penaliza errores grandes. Mismas unidades que $Y$. | $[0, +\infty)$ |
| **MAE** | $\frac{1}{n}\sum_{t=1}^n |Y_t - \hat{Y}_t|$ | Error absoluto medio. Más robusta a outliers que RMSE. Mismas unidades que $Y$. | $[0, +\infty)$ |
| **MAPE** | $\frac{100}{n}\sum_{t=1}^n \left|\frac{Y_t - \hat{Y}_t}{Y_t}\right|$ | Error porcentual. Interpretable como porcentaje promedio de error. Problemático si $Y_t \approx 0$. | $[0, +\infty)$% |
| **SMAPE** | $\frac{100}{n}\sum_{t=1}^n \frac{2|Y_t - \hat{Y}_t|}{|Y_t| + |\hat{Y}_t|}$ | MAPE simétrico. Evita la asimetría del MAPE (no penaliza más sobreestimaciones que subestimaciones). | $[0, 200]$% |
| **R²** | $1 - \frac{\sum(Y_t - \hat{Y}_t)^2}{\sum(Y_t - \bar{Y})^2}$ | Proporción de varianza explicada. Comparable a un modelo naive (media constante). | $(-\infty, 1]$ |
| **MaxAE** | $\max_t |Y_t - \hat{Y}_t|$ | Peor error puntual. Útil para gestión de riesgo. | $[0, +\infty)$ |

**Umbrales de referencia para MAPE en pronóstico de precios:**

| MAPE | Calidad del pronóstico |
|---|---|
| < 5% | Excelente |
| 5–10% | Bueno |
| 10–20% | Razonable |
| 20–50% | Deficiente |
| > 50% | Muy deficiente |

---

### 4.9 Sección 8: Validación Cruzada Rolling

```python
def validacion_rolling(Y, X, order, seasonal_order, n_initial, step=1):
```

#### ¿Por qué no usar validación cruzada estándar k-fold?

La validación cruzada k-fold estándar asigna particiones aleatorias de los datos a train y test. En series temporales, esto viola la dependencia temporal: se usarían datos futuros para predecir el pasado, produciendo métricas optimistas irreales.

#### Rolling Forecast con Ventana Expandible

```
Iteración 1:  TRAIN = [t₁, ..., t_n]           TEST = [t_{n+1}]
Iteración 2:  TRAIN = [t₁, ..., t_n, t_{n+1}]  TEST = [t_{n+2}]
Iteración 3:  TRAIN = [t₁, ..., t_{n+2}]        TEST = [t_{n+3}]
...
```

Esta estrategia:
- **Respeta el orden temporal** (nunca usa información futura).
- **Simula el proceso real de pronóstico**: el modelo se re-estima con todos los datos disponibles en cada período.
- **Produce un estimador más robusto** del error de generalización que una única partición train/test.
- **Revela el comportamiento del error** a lo largo del tiempo (si el error crece hacia el final, el modelo se degrada con el tiempo → señal de que los datos recientes son más difíciles de predecir).

**`n_initial`** se establece en el 75% de los datos de entrenamiento, garantizando que el modelo tenga suficientes datos para producir predicciones estables antes de comenzar la validación.

---

### 4.10 Sección 9: Gráficos de Resultados

Ver la sección completa de descripción de gráficos en [Sección 5](#5-descripción-de-los-gráficos-generados).

---

### 4.11 Sección 10: Exportación de Resultados

#### Función `guardar_resultados_csv`

Genera tres archivos CSV:

**1. `pronostico_futuro_cemento.csv`**

| Columna | Descripción |
|---|---|
| `Fecha` | Mes del pronóstico (formato YYYY-MM-DD, inicio de mes) |
| `Media` | Precio predicho (media de la distribución predictiva) |
| `IC_Inferior` | Límite inferior del intervalo de confianza al 95% |
| `IC_Superior` | Límite superior del intervalo de confianza al 95% |
| `Nivel_Rio_Exog` | Nivel del río usado como exógena en ese mes |

**2. `prediccion_conjunto_prueba.csv`**

| Columna | Descripción |
|---|---|
| `Fecha` | Mes del período de prueba |
| `Precio_Real` | Precio real observado |
| `Precio_Predicho` | Precio predicho por el modelo |
| `Error_Absoluto` | Real − Predicho |
| `Error_Porcentual_pct` | |Error_Absoluto / Real| × 100 |
| `Error_Relativo_pct` | Error_Absoluto / Real × 100 (con signo) |

**3. `metricas_modelo.csv`**

Una fila con todas las métricas: RMSE, MAE, MAPE, SMAPE, R², MaxAE.

#### Función `generar_informe_texto`

El informe `informe_sarimax.txt` incluye:
1. Resumen del dataset (estadísticas descriptivas).
2. Parámetros óptimos del modelo (orden ARIMA y estacional).
3. Métricas de evaluación del conjunto de prueba.
4. Métricas de validación cruzada rolling.
5. Tests estadísticos sobre residuos (Ljung-Box, Shapiro-Wilk).
6. Tabla completa del pronóstico futuro.
7. Resumen estadístico del modelo (`statsmodels.summary()`).

---

### 4.12 Sección 11: Pipeline Principal (main)

La función `main()` orquesta los 12 pasos del pipeline en orden secuencial, pasando los resultados de cada paso como entrada al siguiente. Retorna un diccionario con los objetos principales:

```python
return {
    "modelo"          : final_res,      # Objeto SARIMAXResults de statsmodels
    "forecast"        : forecast_df,    # DataFrame con el pronóstico futuro
    "metricas_test"   : metricas_test,  # Dict con métricas del test set
    "metricas_rolling": metricas_rolling,
    "parametros"      : {
        "order"         : best_order,
        "seasonal_order": best_seasonal_order,
    },
}
```

---

## 5. Descripción de los Gráficos Generados

### Gráfico 01: `01_series_temporales.png`
**Título:** Series Temporales: Precio del Cemento, Nivel del Río e Indicador COVID-19

**¿Qué muestra?**
Tres paneles apilados verticalmente muestran la evolución temporal de cada variable. El panel de cuarentena clarifica qué período fue afectado por restricciones COVID-19.

**¿Qué observar para la tesis?**
- ¿Hay una tendencia creciente en el precio del cemento? → Justifica la diferenciación `d≥1`.
- ¿Se observan picos/valles recurrentes en el mismo período cada año? → Confirma la necesidad de componente estacional.
- ¿Los movimientos del nivel del río parecen anticipar o coincidir con movimientos en el precio? → Justifica el uso del nivel del río como variable exógena.

---

### Gráfico 02: `02_descomposicion_estacional.png`
**Título:** Descomposición Estacional Aditiva — Período m=12 meses

**¿Qué muestra?**
La descomposición **aditiva** separa la serie en:
- **Tendencia:** Movimiento de largo plazo (extraído por media móvil de orden `m`).
- **Estacionalidad:** Patrón que se repite cada 12 meses.
- **Residuos:** Componente irregular que no captura la tendencia ni la estacionalidad.

**¿Qué observar?**
- La estacionalidad debe mostrar un patrón visual claro y repetitivo.
- Los residuos deben parecerse a ruido blanco (sin estructura visible). Si tienen estructura, el modelo SARIMA necesitará más parámetros.
- Si la amplitud de la estacionalidad crece con el nivel de la serie, se debería usar descomposición **multiplicativa** en su lugar.

**Modelo aditivo vs. multiplicativo:**

| | Aditivo | Multiplicativo |
|---|---|---|
| Supuesto | $Y_t = T_t + S_t + R_t$ | $Y_t = T_t \times S_t \times R_t$ |
| Cuándo usar | Amplitud estacional constante | Amplitud proporcional al nivel |
| Transformación alternativa | — | Aplicar log: $\log(Y_t)$ y usar aditivo |

---

### Gráfico 03: `03_acf_pacf.png`
**Título:** Función de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)

**¿Qué muestra?**
La ACF mide la correlación de la serie con sus propios rezagos $k$:
$$\rho(k) = \frac{\text{Cov}(Y_t, Y_{t-k})}{\text{Var}(Y_t)}$$

La PACF mide la correlación con el rezago $k$ **eliminando** el efecto de los rezagos intermedios.

**Guía de identificación de órdenes:**

| Patrón ACF | Patrón PACF | Modelo sugerido |
|---|---|---|
| Decae exponencialmente | Corte abrupto en $k=p$ | AR($p$) |
| Corte abrupto en $k=q$ | Decae exponencialmente | MA($q$) |
| Ambos decaen exponencialmente | Ambos decaen | ARMA($p$,$q$) |
| Picos en rezagos $m, 2m, 3m$... | Ídem | Componente estacional |

Las líneas punteadas azules representan las **bandas de confianza al 95%** ($\pm 1.96/\sqrt{T}$). Barras dentro de las bandas son estadísticamente indistinguibles de cero.

---

### Gráfico 04: `04_correlacion.png`
**Título:** Análisis de Correlación entre Variables

**Panel izquierdo:** Mapa de calor de la matriz de correlación de Pearson.
**Panel derecho:** Diagrama de dispersión precio vs. nivel del río, coloreado por índice temporal.

**¿Qué observar?**
- La correlación entre `Precio_Cemento` y `Nivel_Rio` cuantifica la relación lineal. Un coeficiente negativo fuerte indica que a mayor nivel del río (mejor navegabilidad → menor costo logístico), el precio tiende a bajar.
- El color del scatter por índice temporal revela si la relación ha cambiado en el tiempo (no-estacionariedad de la relación).

---

### Gráfico 05: `05_patron_estacional.png`
**Título:** Patrones Estacionales del Precio del Cemento

**Panel izquierdo:** Boxplot del precio según el mes del año.
**Panel derecho:** Precio promedio anual con banda de ±1 desviación estándar.

**¿Qué observar?**
- En el boxplot: ¿Hay meses sistemáticamente más caros o más baratos? (ej. mayor precio en temporada seca cuando hay más construcción).
- En el gráfico anual: ¿La tendencia es creciente? ¿Hay quiebres estructurales (COVID-19)?

---

### Gráfico 06: `06_pronostico_completo.png`
**Título:** Pronóstico del Precio del Cemento — Modelo SARIMAX

**Panel superior:** Serie completa desde el inicio hasta el final del pronóstico.
**Panel inferior:** Zoom en los últimos 48 meses de entrenamiento + período de prueba + pronóstico.

**Colores:**
- Azul oscuro: Datos de entrenamiento.
- Naranja: Datos reales del período de prueba (los que el modelo "no vio").
- Verde discontinuo: Predicción sobre el período de prueba.
- Rojo discontinuo: Pronóstico futuro.
- Rosa translúcido: Intervalo de confianza al 95%.

**¿Qué observar?**
- La predicción verde debe seguir de cerca los puntos naranjas → validación exitosa.
- El pronóstico rojo debe continuar la tendencia de forma plausible.
- Las bandas del IC deben ensancharse conforme avanza el horizonte de predicción.

---

### Gráfico 07: `07_diagnostico_residuos.png`
**Título:** Diagnóstico Completo de Residuos — Modelo SARIMAX Final

Este es el gráfico diagnóstico más importante para la validez estadística del modelo.

**Panel 1 — Residuos en el tiempo:**
Los residuos $\hat{\varepsilon}_t = Y_t - \hat{Y}_t$ deben:
- Fluctuar aleatoriamente alrededor de cero.
- No mostrar tendencia, estacionalidad ni heteroscedasticidad.

**Panel 2 — Histograma vs. Normal teórica:**
Evalúa la normalidad de los residuos. La distribución real (barras azules) debe aproximarse a la curva normal teórica (línea roja). Desviaciones fuertes en las colas indican presencia de outliers o distribución con colas pesadas.

**Panel 3 — Q-Q Plot:**
Compara los cuantiles empíricos de los residuos con los cuantiles de la distribución normal teórica. Si los puntos caen sobre la línea roja, los residuos son normales. El $R^2$ del Q-Q plot cuantifica la bondad de ajuste a la normalidad.

**Panel 4 — ACF de residuos:**
Si el modelo está bien especificado, los residuos deben ser **ruido blanco**: sin autocorrelación significativa a ningún rezago. Barras fuera de las bandas de confianza indican estructura no capturada → se necesita un modelo con más parámetros.

**Panel 5 — Residuos estandarizados vs. valores ajustados:**
Evalúa la **homocedasticidad** (varianza constante). Los puntos deben distribuirse uniformemente alrededor de cero sin formar un patrón de embudo (que indicaría heterocedasticidad: mayor error en valores altos que bajos).

---

### Gráfico 08: `08_comparacion_real_predicho.png`
**Título:** Análisis del Conjunto de Prueba — Real vs Predicho

Cuatro paneles que analizan exhaustivamente el error en el período de prueba (los 15 meses no vistos por el modelo).

**Panel superior izquierdo:** Superposición de líneas real (naranja) vs. predicho (verde). Muestra cuán cerca sigue la predicción al valor real.
**Panel superior derecho:** Scatter de predicho vs. real. Los puntos deben estar cerca de la línea de 45° (predicción perfecta).
**Panel inferior izquierdo:** Error absoluto por mes. Barras verdes = sobreestimación, barras rojas = subestimación. La línea azul muestra el error medio (idealmente ≈ 0).
**Panel inferior derecho:** MAPE por mes (%). Muestra si el error porcentual es uniforme o si hay meses específicamente difíciles de predecir.

---

### Gráfico 09: `09_pronostico_intervalo_confianza.png`
**Título:** Pronóstico Futuro con Intervalos de Confianza

**Panel izquierdo:** Contexto histórico reciente (últimos 48 meses) seguido del pronóstico con IC 80% y IC 95%.
**Panel derecho:** Detalle del pronóstico con anotación de valores numéricos cada 2 meses.

**¿Cómo se calculan los intervalos de confianza?**

El IC al $(1-\alpha)\%$ para un pronóstico $h$ pasos adelante es:
$$\hat{Y}_{T+h} \pm z_{\alpha/2} \cdot \sqrt{\hat{\sigma}^2 \left(1 + \sum_{j=0}^{h-1} \hat{\psi}_j^2\right)}$$

donde $\hat{\psi}_j$ son los coeficientes de la representación MA($\infty$) del modelo. A medida que $h$ crece, más términos $\hat{\psi}_j^2 > 0$ se suman, **ensanchando** el intervalo. Esto refleja correctamente la acumulación de incertidumbre en el pronóstico multi-paso.

---

### Gráfico 10: `10_rolling_forecast_cv.png`
**Título:** Validación Cruzada con Rolling Forecast

**Panel superior:** Real vs. predicho con un paso adelante en cada iteración.
**Panel inferior:** Error del rolling forecast, distinguiendo sobreestimaciones (verde) de subestimaciones (rojo).

Este gráfico es la representación visual de la robustez del modelo. Si el error es consistentemente pequeño y aleatorio a lo largo del tiempo, el modelo es estable y generalizable.

---

### Gráfico 11: `11_ajuste_insample.png`
**Título:** Ajuste In-Sample del Modelo SARIMAX Final

Muestra los **valores ajustados** (predichos dentro de los datos de entrenamiento) superpuestos con la serie real. Una diferencia importante con el gráfico 06: aquí se muestra todo el histórico, incluyendo datos que el modelo "vio" durante el entrenamiento.

**Advertencia:** Un ajuste in-sample perfecto no garantiza un buen pronóstico out-of-sample (sobreajuste). Por eso es imprescindible el gráfico 08 (evaluación out-of-sample).

---

## 6. Descripción de los Archivos de Salida

### `/kaggle/working/pronostico_futuro_cemento.csv`

Contiene el pronóstico mensual del precio del cemento para el período futuro. Es el resultado principal de la investigación.

**Ejemplo de contenido:**

```
Fecha,Media,IC_Inferior,IC_Superior,Nivel_Rio_Exog
2024-01-01,1850.32,1742.15,1958.49,3.21
2024-02-01,1873.41,1721.83,2025.00,3.45
...
```

**Uso en la tesis:** Esta tabla se puede incluir directamente en el capítulo de resultados como evidencia de las predicciones del modelo.

### `/kaggle/working/prediccion_conjunto_prueba.csv`

Permite al lector de la tesis verificar la precisión del modelo período a período.

### `/kaggle/working/metricas_modelo.csv`

Una sola fila con todas las métricas. Facilita la comparación con otros modelos (ej. ARIMA sin exógenas, Prophet, LSTM) en una tabla comparativa de la tesis.

### `/kaggle/working/informe_sarimax.txt`

Documento de texto completo con toda la información del modelo. El resumen estadístico de `statsmodels` incluye:
- Coeficientes estimados con errores estándar y p-valores.
- Tests de diagnóstico (Ljung-Box, Jarque-Bera, heteroscedasticidad).
- Criterios de información AIC y BIC.

---

## 7. Métricas de Evaluación: Interpretación Profunda

### RMSE vs. MAE: ¿Cuál usar?

El RMSE es más sensible a errores grandes debido al cuadrado. En pronóstico de precios:
- Si un error de predicción grande (ej. subestimar el precio en 500 pesos) tiene consecuencias económicas graves → priorizar **RMSE**.
- Si todos los errores tienen igual costo independientemente de su magnitud → priorizar **MAE**.

**Relación entre RMSE y MAE:** Si los errores siguen una distribución normal, $\text{RMSE} = \sqrt{\pi/2} \cdot \text{MAE} \approx 1.25 \cdot \text{MAE}$. Si $\text{RMSE} \gg 1.25 \cdot \text{MAE}$, hay outliers en los errores.

### Interpretación de R²

- $R^2 = 1$: Predicción perfecta.
- $R^2 = 0$: El modelo no mejora la predicción naive (usar la media histórica).
- $R^2 < 0$: El modelo es **peor** que predecir siempre la media. Indica sobreajuste severo o modelo mal especificado.

**Advertencia:** Un $R^2$ alto en datos de entrenamiento (in-sample) no garantiza buen desempeño predictivo. El $R^2$ del conjunto de prueba (out-of-sample) es el indicador relevante para la tesis.

---

## 8. Tests Estadísticos: Interpretación

### Test de Ljung-Box

**Hipótesis:**
- $H_0$: Los residuos son ruido blanco (no hay autocorrelación hasta el rezago $K$).
- $H_1$: Existe autocorrelación significativa en algún rezago hasta $K$.

**Estadístico:**
$$Q(K) = n(n+2)\sum_{k=1}^K \frac{\hat{\rho}_k^2}{n-k}$$

**Regla:** Si $p > 0.05$ para rezagos $K=10$ y $K=20$, se acepta $H_0$ → los residuos son ruido blanco → el modelo capturó toda la estructura de la serie.

### Test de Shapiro-Wilk

**Hipótesis:**
- $H_0$: Los residuos provienen de una distribución normal.
- $H_1$: Los residuos NO son normales.

**Regla:** Si $p > 0.05$, no se rechaza la normalidad. La normalidad de los residuos es importante para que los intervalos de confianza sean válidos.

**Nota práctica:** Con series largas ($n > 50$), los tests de normalidad son muy sensibles y rechazan $H_0$ incluso con desviaciones pequeñas e irrelevantes prácticamente. Combinar siempre con el Q-Q plot y el histograma para una evaluación visual.

---

## 9. Rol de la GPU en el Pipeline

### Resumen ejecutivo

| Componente | ¿Beneficia de GPU? | Alternativa GPU |
|---|---|---|
| SARIMAX (L-BFGS) | ❌ No (CPU-bound) | — |
| auto_arima (búsqueda) | ⚠️ Parcialmente (más núcleos CPU) | — |
| Carga de datos (pandas) | ❌ No | cuDF (RAPIDS) |
| Cálculo de métricas (NumPy) | ✅ Sí (si cuPy disponible) | cuPy |
| Generación de gráficos | ❌ No | — |

### ¿Cuándo usar el acelerador GPU en Kaggle para este proyecto?

**Usar GPU si:**
1. El dataset es muy grande (>10,000 observaciones) y las operaciones de preprocesamiento son el cuello de botella.
2. Se agregan modelos de comparación basados en Deep Learning (LSTM, GRU, Transformer).
3. Se realiza una búsqueda por grilla masiva de parámetros (desactivar `stepwise=True`).

**No hay ventaja en activar GPU si** el único objetivo es entrenar el modelo SARIMAX con el dataset actual (datos mensuales con ~120-200 observaciones).

### Habilitación práctica en Kaggle

En el notebook de Kaggle, activar GPU: Settings → Accelerator → GPU T4 x2.

En el script Python, el hardware se detecta automáticamente con `verificar_hardware()`.

---

## 10. Configuración de Rutas en Kaggle

### Paso a paso para ejecutar en Kaggle

**1. Crear el dataset en Kaggle:**
   - Ir a `kaggle.com/your-username/datasets` → New Dataset.
   - Nombre del dataset: `dataset-sarimax-cemento`.
   - Subir los archivos:
     - `precios_cemento_interpolado.csv`
     - `promedio_mensual_nivel_predicho.csv`

**2. Crear un nuevo Notebook en Kaggle:**
   - Ir a Kaggle → New Notebook → Script (`.py`).
   - En la sección "Data" → Add Data → buscar `dataset-sarimax-cemento`.

**3. Agregar el script:**
   - Copiar el contenido de `prediccion_cemento_sarimax.py` al editor.
   - O subir el archivo directamente.

**4. Configurar el hardware:**
   - Settings → Accelerator → GPU T4 (opcional, ver Sección 9).
   - Internet → On (para instalar pmdarima si no está disponible).

**5. Ejecutar:**
   - Session Options → Run All.
   - Los archivos de salida aparecen en el panel "Output" a la derecha.

**Estructura de rutas verificada:**

```python
# Archivos de entrada (READ-ONLY)
/kaggle/input/dataset-sarimax-cemento/precios_cemento_interpolado.csv
/kaggle/input/dataset-sarimax-cemento/promedio_mensual_nivel_predicho.csv

# Archivos de salida (ESCRITURA)
/kaggle/working/pronostico_futuro_cemento.csv    ← Resultado principal
/kaggle/working/prediccion_conjunto_prueba.csv
/kaggle/working/metricas_modelo.csv
/kaggle/working/informe_sarimax.txt
/kaggle/working/graficos/01_series_temporales.png
/kaggle/working/graficos/02_descomposicion_estacional.png
/kaggle/working/graficos/03_acf_pacf.png
/kaggle/working/graficos/04_correlacion.png
/kaggle/working/graficos/05_patron_estacional.png
/kaggle/working/graficos/06_pronostico_completo.png
/kaggle/working/graficos/07_diagnostico_residuos.png
/kaggle/working/graficos/08_comparacion_real_predicho.png
/kaggle/working/graficos/09_pronostico_intervalo_confianza.png
/kaggle/working/graficos/10_rolling_forecast_cv.png
/kaggle/working/graficos/11_ajuste_insample.png
```

---

## 11. Guía de Interpretación para la Tesis

### Capítulo de Metodología

Para la sección metodológica de la tesis, se recomienda:

1. **Describir el preprocesamiento:** Mencionar la interpolación de datos faltantes, la conversión de frecuencia a mensual y la selección de variables.

2. **Justificar el período estacional m=12:** Argumentar desde la naturaleza económica del problema (ciclo anual de construcción, estacionalidad del río).

3. **Describir el proceso de selección de parámetros:** Explicar el algoritmo stepwise de auto_arima y el criterio AIC.

4. **Justificar la partición train/test:** Los últimos 15 meses como conjunto de prueba representan aproximadamente el 10-15% de los datos, siguiendo la convención en series temporales.

5. **Describir la validación rolling:** Enfatizar que respeta el orden temporal y simula el proceso real de pronóstico.

### Capítulo de Resultados

Para presentar los resultados:

1. **Tabla de parámetros del modelo:** SARIMAX(p,d,q)(P,D,Q)₁₂ con los valores obtenidos.

2. **Tabla de métricas:** RMSE, MAE, MAPE, SMAPE, R² del conjunto de prueba Y del rolling CV.

3. **Gráfico principal:** Usar el Gráfico 06 (pronóstico completo) como figura central del capítulo.

4. **Diagnóstico de residuos:** El Gráfico 07 demuestra que el modelo es estadísticamente válido.

5. **Tabla de pronóstico:** El contenido de `pronostico_futuro_cemento.csv` directamente en el texto.

### Comparación con literatura

Para contextualizar los resultados:

| Indicador | Este modelo | Referencia típica en literatura |
|---|---|---|
| MAPE | Valor obtenido | 5-15% para pronóstico de precios de materiales |
| R² | Valor obtenido | > 0.80 considerado bueno en series de precios |
| AIC | Valor obtenido | Menor es mejor; comparar con ARIMA sin exógenas |

---

## 12. Referencias Bibliográficas

1. **Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.** (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

2. **Hyndman, R.J., & Athanasopoulos, G.** (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. Disponible en: https://otexts.com/fpp3/

3. **Hamilton, J.D.** (1994). *Time Series Analysis*. Princeton University Press.

4. **Shumway, R.H., & Stoffer, D.S.** (2017). *Time Series Analysis and Its Applications: With R Examples* (4th ed.). Springer.

5. **Hyndman, R.J., & Khandakar, Y.** (2008). Automatic time series forecasting: The forecast package for R. *Journal of Statistical Software*, 27(3), 1–22.

6. **Akaike, H.** (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.

7. **Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y.** (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.

8. **Ljung, G.M., & Box, G.E.P.** (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297–303.

9. **Dickey, D.A., & Fuller, W.A.** (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427–431.

10. **Smith, T.G.** (2017). pmdarima: ARIMA estimators for Python. Disponible en: http://www.alkaline-ml.com/pmdarima

11. **Seabold, S., & Perktold, J.** (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference* (SciPy 2010).

---

*Documentación generada para el proyecto de tesis: "Predicción del Precio del Cemento mediante Modelo SARIMAX con Variables Exógenas Hidrológicas"*

*Archivo: `DOCUMENTACION_SARIMAX.md` | Versión: 1.0 | Fecha: 2026*
