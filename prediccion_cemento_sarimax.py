# =============================================================================
# prediccion_cemento_sarimax.py
# =============================================================================
# MODELO SARIMAX - PREDICCIÓN DEL PRECIO DEL CEMENTO
# Descripción: Pipeline de series temporales con SARIMAX,
#              gráficos para tesis e informes en CSV.
#
# Archivos de entrada requeridos (en /kaggle/input/dataset-sarimax-cemento/):
#   - precios_cemento_interpolado.csv   : Serie histórica del precio del cemento
#                                         más indicador binario de cuarentena COVID-19.
#   - nivel_rio_minimo_mensual.csv      : Nivel mínimo mensual del río.
# =============================================================================

# =============================================================================
# SECCIÓN 1: IMPORTACIONES
# =============================================================================

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pmdarima"])

import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima

from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN GLOBAL
# =============================================================================

# ── Rutas de Kaggle ──────────────────────────────────────────────────────────
RUTA_ENTRADA       = "/kaggle/input/"
RUTA_SALIDA        = "/kaggle/working/"
NOMBRE_DATASET     = "dataset-sarimax-cemento"

RUTA_CEMENTO       = os.path.join(RUTA_ENTRADA, NOMBRE_DATASET,
                                   "precios_cemento_interpolado.csv")
RUTA_NIVEL_RIO     = os.path.join(RUTA_ENTRADA, NOMBRE_DATASET,
                                   "nivel_rio_minimo_mensual.csv")

# ── Directorio de salida para gráficos ──────────────────────────────────────
DIR_GRAFICOS = os.path.join(RUTA_SALIDA, "graficos")
os.makedirs(DIR_GRAFICOS, exist_ok=True)

# ── Parámetros del modelo ─────────────────────────────────────────────────────
PERIODO_ESTACIONAL = 12     # Mensual → m = 12
ALPHA_IC           = 0.05   # Nivel de significancia para intervalos de confianza (95 %)

# ── Partición de datos 70 / 15 / 15 ──────────────────────────────────────────────
#
# Se utiliza la MISMA partición que el modelo LSTM para garantizar una
# comparación justa entre ambos modelos.
#
# En LSTM:
#   - 70% Train      → entrenar los pesos de la red neuronal
#   - 15% Validación → early stopping y ajuste de hiperparámetros
#   - 15% Test       → evaluación final (datos nunca vistos)
#
# En SARIMAX (este script):
#   - 70% Train      → auto_arima selecciona (p,d,q)(P,D,Q,m) por AIC
#                       y entrena el modelo con esos parámetros
#   - 15% Validación → el modelo entrenado predice este período para
#                       obtener el RMSE de validación (análogo al
#                       validation loss del LSTM)
#   - 15% Test       → se re-entrena con train+val (85%) y se evalúa
#                       sobre test para obtener el RMSE final
#
# Nota: tradicionalmente SARIMAX no necesita un set de validación separado
# porque la selección de parámetros se hace via AIC (criterio de información
# que penaliza la complejidad del modelo). Sin embargo, al usar la misma
# partición que LSTM, se asegura que ambos modelos fueron evaluados sobre
# exactamente los mismos datos, lo cual es requisito para una comparación
# válida en la tesis.
# ───────────────────────────────────────────────────────────────────────────
RATIO_TRAIN = 0.70
RATIO_VAL   = 0.15
RATIO_TEST  = 0.15

# ── Paleta de colores VIVOS para presentación ────────────────────────────────
COLORES = {
    "train"    : "#2979FF",   # Azul eléctrico brillante
    "test"     : "#FF6D00",   # Naranja intenso
    "fit"      : "#00E676",   # Verde neón
    "forecast" : "#FF1744",   # Rojo vibrante
    "ci_95"    : "#FF8A80",   # Rojo claro para bandas
    "nivel"    : "#AA00FF",   # Violeta brillante
    "covid"    : "#FF1744",   # Rojo vivo
    "neutral"  : "#78909C",   # Gris azulado (solo para residuos)
    "pos_err"  : "#00E676",   # Verde neón
    "neg_err"  : "#FF1744",   # Rojo vibrante
}

# ── Estilo de gráficos ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"       : 150,
    "savefig.dpi"      : 200,
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.titlesize"   : 12,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 10,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 9,
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#FAFAFA",
    "axes.grid"        : True,
    "grid.alpha"       : 0.35,
    "grid.linestyle"   : "--",
})


# =============================================================================
# SECCIÓN 3: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def _limpiar_columna_numerica(serie: pd.Series) -> pd.Series:
    """Convierte una columna que puede tener comas decimales a float64."""
    return serie.astype(str).str.replace(",", ".", regex=False).astype(float)


def cargar_datos_historicos(ruta: str) -> pd.DataFrame:
    """
    Carga el CSV de precios de cemento interpolados.
    Aplica renombrado, conversión de tipos y frecuencia mensual.
    """
    print(f"\n{'━'*60}")
    print("  PASO 1 ▸ Cargando datos históricos")
    print(f"{'━'*60}")
    print(f"  Archivo: {ruta}")

    df = pd.read_csv(ruta, index_col="Fecha", parse_dates=True, decimal=",")

    # Mapeo flexible de nombres de columnas
    rename = {}

    # Precio: seleccionar UNA sola columna con prioridad definida
    precio_col = None
    precio_prioridad = [
        "Precio_Promedio_Polinomial_2",
        "Precio_Cemento",
        "Precio_Promedio",
    ]
    for candidata in precio_prioridad:
        if candidata in df.columns:
            precio_col = candidata
            break
    if precio_col is None:
        for c in df.columns:
            if "precio" in c.lower():
                precio_col = c
                break
    if precio_col and precio_col != "Precio_Cemento":
        rename[precio_col] = "Precio_Cemento"

    # Nivel y Cuarentena
    for c in df.columns:
        cl = c.lower()
        if "nivel" in cl and "Nivel_Rio" not in rename.values():
            rename[c] = "Nivel_Rio"
        elif ("cuarentena" in cl or "covid" in cl) and "Cuarentena" not in rename.values():
            rename[c] = "Cuarentena"
    df = df.rename(columns=rename)

    # Conversión de tipos
    for col in ["Precio_Cemento", "Nivel_Rio"]:
        if col in df.columns:
            df[col] = _limpiar_columna_numerica(df[col])

    if "Cuarentena" in df.columns:
        df["Cuarentena"] = (
            pd.to_numeric(df["Cuarentena"], errors="coerce").fillna(0).astype(int)
        )

    # Seleccionar columnas necesarias
    cols_necesarias = [c for c in ["Precio_Cemento", "Nivel_Rio", "Cuarentena"]
                       if c in df.columns]
    df = df[cols_necesarias]

    # Frecuencia mensual
    df = df.asfreq("MS")

    # Manejo de valores faltantes
    n_nan = df.isnull().sum().sum()
    if n_nan > 0:
        print(f"  ⚠  {n_nan} valores faltantes detectados → interpolación lineal.")
        df = df.interpolate(method="linear").bfill().ffill()

    print(f"  ✓  {len(df)} observaciones | "
          f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")
    print(f"\n  Estadísticas descriptivas:")
    print(df.describe().round(4).to_string(index=True))
    print()

    return df


def cargar_nivel_rio_completo(ruta: str) -> pd.DataFrame:
    """
    Carga el CSV completo del nivel mínimo mensual del río.
    Contiene período histórico + futuro (usado como exógena en pronóstico).
    """
    print(f"\n{'━'*60}")
    print("  PASO 2 ▸ Cargando datos del nivel del río (histórico + futuro)")
    print(f"{'━'*60}")
    print(f"  Archivo: {ruta}")

    df = pd.read_csv(ruta, index_col="Fecha", parse_dates=True, decimal=",")

    for c in df.columns:
        if "nivel" in c.lower():
            df = df.rename(columns={c: "Nivel_Rio"})
            break

    df["Nivel_Rio"] = _limpiar_columna_numerica(df["Nivel_Rio"])
    df = df[["Nivel_Rio"]]
    df = df.asfreq("MS")

    n_nan = df["Nivel_Rio"].isnull().sum()
    if n_nan > 0:
        print(f"  ⚠  {n_nan} valores faltantes → interpolación lineal.")
        df["Nivel_Rio"] = df["Nivel_Rio"].interpolate(method="linear").bfill().ffill()

    print(f"  ✓  {len(df)} registros | "
          f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")
    print()

    return df


# =============================================================================
# SECCIÓN 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

def test_estacionariedad(serie: pd.Series, nombre: str = "Serie") -> dict:
    """
    Ejecuta los tests ADF y KPSS para evaluar estacionariedad.

    ADF: H₀ = serie NO estacionaria → p < 0.05 rechaza → estacionaria.
    KPSS: H₀ = serie ES estacionaria → p < 0.05 rechaza → NO estacionaria.
    """
    res = {}
    s = serie.dropna()

    adf = adfuller(s, autolag="AIC")
    res["adf_stat"]        = adf[0]
    res["adf_pvalue"]      = adf[1]
    res["adf_estacionaria"] = adf[1] < 0.05

    try:
        kpss_r = kpss(s, regression="ct", nlags="auto")
        res["kpss_stat"]        = kpss_r[0]
        res["kpss_pvalue"]      = kpss_r[1]
        res["kpss_estacionaria"] = kpss_r[1] > 0.05
    except Exception:
        res["kpss_stat"] = res["kpss_pvalue"] = res["kpss_estacionaria"] = None

    print(f"\n  Test de Estacionariedad → {nombre}")
    print(f"    ADF  | estadístico: {res['adf_stat']:8.4f}  p-valor: {res['adf_pvalue']:.4f}  "
          f"→ {'ESTACIONARIA ✓' if res['adf_estacionaria'] else 'NO ESTACIONARIA ✗'}")
    if res["kpss_stat"] is not None:
        print(f"    KPSS | estadístico: {res['kpss_stat']:8.4f}  p-valor: {res['kpss_pvalue']:.4f}  "
              f"→ {'ESTACIONARIA ✓' if res['kpss_estacionaria'] else 'NO ESTACIONARIA ✗'}")

    return res


# ──────────────────────────────────────────────────────────────────────────────
#  GRÁFICOS EDA
# ──────────────────────────────────────────────────────────────────────────────

def grafico_01_series_temporales(df: pd.DataFrame, dir_salida: str) -> str:
    """Gráfico 1: Las tres series temporales apiladas verticalmente."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        "Series Temporales: Precio del Cemento, Nivel del Río e Indicador COVID-19",
        fontsize=13, fontweight="bold", y=0.99,
    )

    ax = axes[0]
    ax.plot(df.index, df["Precio_Cemento"], color=COLORES["train"], lw=1.8)
    ax.fill_between(df.index, df["Precio_Cemento"], alpha=0.08, color=COLORES["train"])
    ax.set_ylabel("Precio ($/bolsa)", fontsize=10)
    ax.set_title("Precio del Cemento", fontsize=11)

    ax = axes[1]
    ax.plot(df.index, df["Nivel_Rio"], color=COLORES["nivel"], lw=1.8)
    ax.fill_between(df.index, df["Nivel_Rio"], alpha=0.08, color=COLORES["nivel"])
    ax.set_ylabel("Nivel (m)", fontsize=10)
    ax.set_title("Nivel del Río (Variable Exógena)", fontsize=11)

    ax = axes[2]
    ax.fill_between(df.index, df["Cuarentena"],
                    alpha=0.65, color=COLORES["covid"], label="Período COVID-19")
    ax.set_ylabel("Indicador (0/1)", fontsize=10)
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_title("Indicador de Cuarentena COVID-19", fontsize=11)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.1, 1.5)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "01_series_temporales.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_02_descomposicion(serie: pd.Series, periodo: int, dir_salida: str) -> str:
    """Gráfico 2: Descomposición aditiva de la serie de precios."""
    dec = seasonal_decompose(serie.dropna(), model="additive", period=periodo)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(
        f"Descomposición Estacional Aditiva — Período m={periodo} meses",
        fontsize=13, fontweight="bold",
    )

    componentes = [
        (dec.observed,  "Serie Observada",      COLORES["train"]),
        (dec.trend,     "Tendencia",             COLORES["nivel"]),
        (dec.seasonal,  "Componente Estacional", COLORES["fit"]),
        (dec.resid,     "Residuos",              COLORES["neutral"]),
    ]

    for ax, (comp, titulo, color) in zip(axes, componentes):
        ax.plot(comp.index, comp.values, color=color, lw=1.4)
        if titulo == "Residuos":
            ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel("Valor", fontsize=9)

    axes[-1].set_xlabel("Fecha", fontsize=10)
    plt.tight_layout()
    ruta = os.path.join(dir_salida, "02_descomposicion_estacional.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_03_acf_pacf(serie: pd.Series, lags: int, dir_salida: str) -> str:
    """Gráfico 3: ACF y PACF de la serie original y su primera diferencia."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Función de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)\n"
        "Serie Original (fila superior) | Primera Diferencia (fila inferior)",
        fontsize=12, fontweight="bold",
    )

    s_orig = serie.dropna()
    s_diff = s_orig.diff().dropna()

    plot_acf(s_orig, lags=lags, ax=axes[0, 0],
             color=COLORES["train"], title="ACF — Serie Original")
    plot_pacf(s_orig, lags=lags, ax=axes[0, 1],
              color=COLORES["train"], title="PACF — Serie Original")
    plot_acf(s_diff, lags=lags, ax=axes[1, 0],
             color=COLORES["test"], title="ACF — Primera Diferencia ∇Yₜ")
    plot_pacf(s_diff, lags=lags, ax=axes[1, 1],
              color=COLORES["test"], title="PACF — Primera Diferencia ∇Yₜ")

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "03_acf_pacf.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_04_correlacion(df: pd.DataFrame, dir_salida: str) -> str:
    """Gráfico 4: Matriz de correlación y dispersión Precio vs Nivel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Análisis de Correlación entre Variables",
                 fontsize=13, fontweight="bold")

    corr = df.corr()
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
        vmin=-1, vmax=1, ax=axes[0], linewidths=0.6,
        cbar_kws={"label": "Correlación de Pearson", "shrink": 0.8},
        annot_kws={"size": 11},
    )
    axes[0].set_title("Matriz de Correlación de Pearson", fontsize=11)

    scatter = axes[1].scatter(
        df["Nivel_Rio"], df["Precio_Cemento"],
        c=np.arange(len(df)), cmap="plasma", alpha=0.75, s=45, zorder=5,
    )
    plt.colorbar(scatter, ax=axes[1], label="Índice temporal (meses)")

    x_val = df["Nivel_Rio"].dropna().values
    y_val = df["Precio_Cemento"].dropna().values
    if len(x_val) == len(y_val):
        z = np.polyfit(x_val, y_val, 1)
        x_line = np.linspace(x_val.min(), x_val.max(), 200)
        axes[1].plot(x_line, np.poly1d(z)(x_line), "r--", lw=2, label="Tendencia lineal")

    axes[1].set_xlabel("Nivel del Río (m)", fontsize=10)
    axes[1].set_ylabel("Precio del Cemento", fontsize=10)
    axes[1].set_title("Precio del Cemento vs Nivel del Río", fontsize=11)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "04_correlacion.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_05_patron_estacional(serie: pd.Series, dir_salida: str) -> str:
    """Gráfico 5: Boxplot por mes y evolución del precio promedio anual."""
    df_t = pd.DataFrame({
        "Precio": serie.values,
        "Mes": serie.index.month,
        "Año": serie.index.year,
    })
    meses_labels = ["Ene","Feb","Mar","Abr","May","Jun",
                    "Jul","Ago","Sep","Oct","Nov","Dic"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Patrones Estacionales del Precio del Cemento",
                 fontsize=13, fontweight="bold")

    data_mes = [df_t.loc[df_t["Mes"] == m, "Precio"].values for m in range(1, 13)]
    bp = axes[0].boxplot(data_mes, labels=meses_labels, patch_artist=True,
                          notch=False, medianprops={"color": "red", "lw": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
        patch.set_alpha(0.7)
    axes[0].set_title("Distribución del Precio por Mes del Año", fontsize=11)
    axes[0].set_xlabel("Mes", fontsize=10)
    axes[0].set_ylabel("Precio del Cemento", fontsize=10)

    anual = df_t.groupby("Año")["Precio"].agg(["mean", "std"]).reset_index()
    axes[1].plot(anual["Año"], anual["mean"], "o-",
                 color=COLORES["train"], lw=2, ms=7)
    axes[1].fill_between(
        anual["Año"],
        anual["mean"] - anual["std"],
        anual["mean"] + anual["std"],
        alpha=0.15, color=COLORES["train"], label="±1 Desviación estándar",
    )
    axes[1].set_title("Precio Promedio Anual (± 1σ)", fontsize=11)
    axes[1].set_xlabel("Año", fontsize=10)
    axes[1].set_ylabel("Precio Promedio", fontsize=10)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "05_patron_estacional.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


# =============================================================================
# SECCIÓN 5: MODELADO SARIMAX
# =============================================================================

def buscar_parametros_auto(
    Y_train: pd.Series,
    X_train: pd.DataFrame,
    m: int = 12,
) -> tuple[tuple, tuple, object]:
    """
    Busca los órdenes ARIMA y estacionales óptimos usando pmdarima.auto_arima.
    Usa el algoritmo stepwise de Hyndman-Khandakar minimizando AIC.
    """
    print(f"\n{'━'*60}")
    print("  PASO 4 ▸ Búsqueda de parámetros óptimos (auto_arima)")
    print(f"{'━'*60}")
    print(f"  Observaciones de entrenamiento : {len(Y_train)}")
    print(f"  Período estacional (m)         : {m}")
    print(f"  Criterio de selección          : AIC\n")

    t0 = time.time()

    modelo_auto = auto_arima(
        Y_train,
        exogenous      = X_train,
        start_p        = 0, max_p = 5,
        start_q        = 0, max_q = 5,
        d              = None,           # Detección automática via test ADF
        start_P        = 0, max_P = 3,
        start_Q        = 0, max_Q = 3,
        D              = None,           # Detección automática via test OCSB
        seasonal       = True,
        m              = m,
        stepwise       = True,
        information_criterion = "aic",
        trace          = True,
        suppress_warnings = True,
        error_action   = "ignore",
    )

    elapsed = time.time() - t0
    order          = modelo_auto.order
    seasonal_order = modelo_auto.seasonal_order

    print(f"\n  ✓ Búsqueda completada en {elapsed:.1f} s")
    print(f"  Mejor orden ARIMA    : {order}")
    print(f"  Mejor orden estacional: {seasonal_order}")
    print(f"  AIC del mejor modelo : {modelo_auto.aic():.4f}")
    print(f"\n  Resumen del modelo auto_arima:\n{modelo_auto.summary()}")

    return order, seasonal_order, modelo_auto


def entrenar_sarimax(
    Y: pd.Series,
    X: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
    nombre: str = "Modelo SARIMAX",
) -> object:
    """
    Entrena un modelo SARIMAX(p,d,q)(P,D,Q,m) con variables exógenas.
    """
    print(f"\n  Entrenando {nombre}...")
    t0 = time.time()

    modelo = SARIMAX(
        Y,
        exog                  = X,
        order                 = order,
        seasonal_order        = seasonal_order,
        enforce_stationarity  = False,
        enforce_invertibility = False,
        freq                  = "MS",
    )

    resultado = modelo.fit(disp=False, method="lbfgs", maxiter=300)
    elapsed = time.time() - t0

    print(f"  ✓ Entrenado en {elapsed:.1f} s")
    print(f"    AIC: {resultado.aic:.4f}  BIC: {resultado.bic:.4f}  "
          f"Log-L: {resultado.llf:.4f}")

    return resultado


# =============================================================================
# SECCIÓN 6: MÉTRICA DE EVALUACIÓN
# =============================================================================

def calcular_rmse(
    y_real: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    nombre: str = "",
) -> float:
    """Calcula RMSE (Raíz del Error Cuadrático Medio)."""
    r = np.asarray(y_real, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(r, p)))

    if nombre:
        print(f"\n  {'─'*52}")
        print(f"  RMSE — {nombre}")
        print(f"  {'─'*52}")
        print(f"  RMSE (Raíz Error Cuadrático Medio) : {rmse:>10.4f}")
        print(f"  {'─'*52}")

    return rmse


# =============================================================================
# SECCIÓN 7: GRÁFICOS DE RESULTADOS DEL MODELO
# =============================================================================

def grafico_06_pronostico_completo(
    Y_train: pd.Series,
    Y_test: pd.Series,
    test_pred: pd.Series,
    forecast_df: pd.DataFrame,
    rmse_test: float,
    dir_salida: str,
) -> str:
    """
    Gráfico 6: Vista completa del pronóstico — datos de entrenamiento,
    conjunto de prueba (real vs predicho) y pronóstico futuro con IC 95%.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 13))
    fig.suptitle(
        "Pronóstico del Precio del Cemento — Modelo SARIMAX\n"
        f"RMSE={rmse_test:.2f}",
        fontsize=13, fontweight="bold",
    )

    for ax_idx, ax in enumerate(axes):
        ax.plot(Y_train.index, Y_train.values,
                color=COLORES["train"], lw=1.5, label="Entrenamiento", zorder=3)
        ax.plot(Y_test.index, Y_test.values,
                color=COLORES["test"], lw=2.2, marker="o", ms=4,
                label="Real (prueba)", zorder=4)
        ax.plot(test_pred.index, test_pred.values,
                color=COLORES["fit"], lw=1.8, ls="--",
                label=f"Predicción prueba (RMSE={rmse_test:.2f})", zorder=4)
        ax.plot(forecast_df.index, forecast_df["Media"],
                color=COLORES["forecast"], lw=2.2, ls="--",
                marker="s", ms=3, label="Pronóstico futuro", zorder=5)
        ax.fill_between(
            forecast_df.index,
            forecast_df["IC_Inferior"], forecast_df["IC_Superior"],
            alpha=0.20, color=COLORES["forecast"], label="IC 95%",
        )
        ax.axvline(Y_test.index[0],    color="gray",  ls=":", lw=1.2, alpha=0.8)
        ax.axvline(forecast_df.index[0], color="black", ls=":", lw=1.2, alpha=0.8)
        ax.set_ylabel("Precio del Cemento", fontsize=10)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        if ax_idx == 0:
            ax.set_title("Vista Completa de la Serie", fontsize=11)
        else:
            x_min = Y_train.index[-min(48, len(Y_train))]
            ax.set_xlim(left=x_min)
            ax.set_title(
                f"Zoom: Últimos {min(48, len(Y_train))} meses de entrenamiento + "
                "período de prueba + pronóstico",
                fontsize=11,
            )
            ax.set_xlabel("Fecha", fontsize=10)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "06_pronostico_completo.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_07_diagnostico_residuos(resultado_modelo, dir_salida: str) -> str:
    """
    Gráfico 7: Panel de diagnóstico de residuos (4 paneles).
    Verifica: estacionariedad, normalidad, no autocorrelación, homocedasticidad.
    """
    resid = resultado_modelo.resid

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("Diagnóstico Completo de Residuos — Modelo SARIMAX Final",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(resid.index, resid.values, color=COLORES["neutral"], lw=0.9)
    ax1.axhline(0, color="red", ls="--", lw=1.2, alpha=0.7)
    ax1.fill_between(resid.index, resid.values, 0, alpha=0.15, color=COLORES["neutral"])
    ax1.set_title("Residuos en el Tiempo", fontsize=11)
    ax1.set_ylabel("Residuo", fontsize=10)

    mu_r, sigma_r = resid.mean(), resid.std()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(resid.dropna(), bins=25, density=True,
             color="#00BCD4", alpha=0.70, edgecolor="white", label="Distribución real")
    x_n = np.linspace(resid.min(), resid.max(), 300)
    ax2.plot(x_n, stats.norm.pdf(x_n, mu_r, sigma_r), "r-", lw=2, label="Normal teórica")
    ax2.set_title("Distribución de Residuos vs Normal Teórica", fontsize=11)
    ax2.set_xlabel("Residuo", fontsize=10)
    ax2.set_ylabel("Densidad", fontsize=10)
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, r_qq) = stats.probplot(resid.dropna(), dist="norm")
    ax3.scatter(osm, osr, color="#00BCD4", alpha=0.6, s=22, zorder=5)
    ax3.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2, label=f"R²={r_qq**2:.4f}")
    ax3.set_title("Q-Q Plot de Residuos (normalidad)", fontsize=11)
    ax3.set_xlabel("Cuantiles teóricos (Normal)", fontsize=10)
    ax3.set_ylabel("Cuantiles observados", fontsize=10)
    ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[2, 0])
    plot_acf(resid.dropna(), lags=30, ax=ax4, color="#2979FF",
             title="ACF de Residuos")

    ax5 = fig.add_subplot(gs[2, 1])
    resid_std = (resid - mu_r) / sigma_r
    fitted = resultado_modelo.fittedvalues
    ax5.scatter(fitted, resid_std, alpha=0.5, color="#00BCD4", s=22, zorder=5)
    ax5.axhline(0, color="red", ls="--", lw=1.2)
    ax5.axhline(2,  color="orange", ls=":", lw=1.2, alpha=0.7, label="±2σ (95%)")
    ax5.axhline(-2, color="orange", ls=":", lw=1.2, alpha=0.7)
    ax5.set_title("Residuos Estandarizados vs Valores Ajustados", fontsize=11)
    ax5.set_xlabel("Valores Ajustados", fontsize=10)
    ax5.set_ylabel("Residuos Estandarizados", fontsize=10)
    ax5.legend(fontsize=9)

    ruta = os.path.join(dir_salida, "07_diagnostico_residuos.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_08_comparacion_prueba(
    Y_test: pd.Series,
    test_pred: pd.Series,
    rmse_test: float,
    dir_salida: str,
) -> str:
    """Gráfico 8: Análisis detallado del conjunto de prueba — real vs predicho."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Análisis del Conjunto de Prueba — Real vs Predicho (RMSE={rmse_test:.2f})",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: Líneas real vs predicho
    ax = axes[0]
    ax.plot(Y_test.index, Y_test.values, "o-", color=COLORES["test"],
            lw=2, ms=5, label="Real")
    ax.plot(test_pred.index, test_pred.values, "s--", color=COLORES["fit"],
            lw=2, ms=5, label="Predicho")
    ax.set_title("Real vs Predicho en el Período de Prueba", fontsize=11)
    ax.set_ylabel("Precio del Cemento", fontsize=10)
    ax.legend(fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Panel 2: Scatter real vs predicho (línea de 45°)
    ax2 = axes[1]
    ax2.scatter(Y_test.values, test_pred.values, color="#00BCD4", alpha=0.8, s=60, zorder=5)
    lim_min = min(Y_test.min(), test_pred.min()) * 0.98
    lim_max = max(Y_test.max(), test_pred.max()) * 1.02
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2, label="Predicción perfecta")
    ax2.set_xlim(lim_min, lim_max)
    ax2.set_ylim(lim_min, lim_max)
    ax2.set_title("Dispersión: Real vs Predicho", fontsize=11)
    ax2.set_xlabel("Precio Real", fontsize=10)
    ax2.set_ylabel("Precio Predicho", fontsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "08_comparacion_real_predicho.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def _grafico_escenario_individual(
    forecast_df: pd.DataFrame,
    Y_hist: pd.Series,
    color_linea: str,
    titulo: str,
    nombre_archivo: str,
    dir_salida: str,
) -> str:
    """Genera un gráfico individual de pronóstico con IC 80% y IC 95%."""
    ic_rango = forecast_df["IC_Superior"] - forecast_df["IC_Inferior"]
    ic80_sup = forecast_df["Media"] + ic_rango * (1.28 / 1.96) / 2
    ic80_inf = forecast_df["Media"] - ic_rango * (1.28 / 1.96) / 2

    n_hist = min(48, len(Y_hist))
    Y_rec  = Y_hist.iloc[-n_hist:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(Y_rec.index, Y_rec.values, color=COLORES["train"], lw=2,
            label=f"Histórico (últimos {n_hist} meses)")
    ax.plot(forecast_df.index, forecast_df["Media"],
            color=color_linea, lw=2.5, ls="--", marker="o", ms=4,
            label="Pronóstico (media)")
    ax.fill_between(forecast_df.index,
                    forecast_df["IC_Inferior"], forecast_df["IC_Superior"],
                    alpha=0.18, color=color_linea, label="IC 95%")
    ax.fill_between(forecast_df.index, ic80_inf, ic80_sup,
                    alpha=0.30, color=color_linea, label="IC 80%")
    ax.axvline(forecast_df.index[0], color="gray", ls=":", lw=1.2, alpha=0.7)
    ax.set_title("Contexto Histórico + Pronóstico Futuro", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel("Precio del Cemento", fontsize=10)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(forecast_df.index, forecast_df["Media"], "o-",
             color=color_linea, lw=2.2, ms=7, label="Pronóstico")
    ax2.fill_between(forecast_df.index,
                     forecast_df["IC_Inferior"], forecast_df["IC_Superior"],
                     alpha=0.20, color=color_linea, label="IC 95%")
    ax2.fill_between(forecast_df.index, ic80_inf, ic80_sup,
                     alpha=0.35, color=color_linea, label="IC 80%")

    for i, (idx, row) in enumerate(forecast_df.iterrows()):
        if i % 2 == 0:
            ax2.annotate(
                f"{row['Media']:,.0f}",
                xy=(idx, row["Media"]),
                xytext=(0, 10), textcoords="offset points",
                fontsize=8, ha="center", color=color_linea, fontweight="bold",
            )

    ax2.set_title("Detalle del Pronóstico Futuro con Valores", fontsize=11)
    ax2.set_xlabel("Fecha", fontsize=10)
    ax2.set_ylabel("Precio del Cemento", fontsize=10)
    ax2.legend(fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, nombre_archivo)
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_09_escenarios_covid(
    forecast_sin: pd.DataFrame,
    forecast_con: pd.DataFrame,
    Y_hist: pd.Series,
    dir_salida: str,
) -> tuple[str, str]:
    """
    Genera DOS gráficos de pronóstico, uno por escenario COVID.
      09a_pronostico_sin_covid.png → Cuarentena = 0 (condiciones normales)
      09b_pronostico_con_covid.png → Cuarentena = 1 (cuarentena activa)
    """
    ruta_sin = _grafico_escenario_individual(
        forecast_df   = forecast_sin,
        Y_hist        = Y_hist,
        color_linea   = "#1565C0",
        titulo        = (
            "Pronóstico del Precio del Cemento — Escenario Sin COVID (Cuarentena=0)\n"
            f"Modelo SARIMAX — Horizonte: {len(forecast_sin)} meses"
        ),
        nombre_archivo = "09a_pronostico_sin_covid.png",
        dir_salida    = dir_salida,
    )

    ruta_con = _grafico_escenario_individual(
        forecast_df   = forecast_con,
        Y_hist        = Y_hist,
        color_linea   = "#B71C1C",
        titulo        = (
            "Pronóstico del Precio del Cemento — Escenario Con COVID (Cuarentena=1)\n"
            f"Modelo SARIMAX — Horizonte: {len(forecast_con)} meses"
        ),
        nombre_archivo = "09b_pronostico_con_covid.png",
        dir_salida    = dir_salida,
    )

    return ruta_sin, ruta_con


# =============================================================================
# SECCIÓN 8: EXPORTACIÓN DE RESULTADOS
# =============================================================================

def guardar_resultados_csv(
    forecast_df: pd.DataFrame,
    forecast_df_covid: pd.DataFrame,
    Y_test: pd.Series,
    test_pred: pd.Series,
    rmse_test: float,
    ruta_salida: str,
) -> dict:
    """
    Guarda tres archivos CSV:
      1. pronostico_futuro_cemento.csv     → Pronóstico (Cuarentena=0)
      2. pronostico_escenarios_covid.csv   → Comparación Sin/Con COVID
      3. prediccion_conjunto_prueba.csv    → Real vs Predicho en test
    """
    rutas = {}

    # 1. Pronóstico futuro Sin COVID
    p1 = os.path.join(ruta_salida, "pronostico_futuro_cemento.csv")
    forecast_df.to_csv(p1, encoding="utf-8")
    rutas["pronostico"] = p1
    print(f"  ✓ Pronóstico (Sin COVID) guardado : {p1}")

    # 2. Escenarios comparados
    df_escenarios = pd.DataFrame({
        "Media_Sin_Covid"    : forecast_df["Media"].values,
        "IC_Inf_Sin_Covid"   : forecast_df["IC_Inferior"].values,
        "IC_Sup_Sin_Covid"   : forecast_df["IC_Superior"].values,
        "Media_Con_Covid"    : forecast_df_covid["Media"].values,
        "IC_Inf_Con_Covid"   : forecast_df_covid["IC_Inferior"].values,
        "IC_Sup_Con_Covid"   : forecast_df_covid["IC_Superior"].values,
        "Diferencia_Covid"   : forecast_df_covid["Media"].values - forecast_df["Media"].values,
        "Nivel_Rio_Exog"     : forecast_df["Nivel_Rio_Exog"].values,
    }, index=forecast_df.index)
    df_escenarios.index.name = "Fecha"
    p2 = os.path.join(ruta_salida, "pronostico_escenarios_covid.csv")
    df_escenarios.to_csv(p2, encoding="utf-8")
    rutas["escenarios"] = p2
    print(f"  ✓ Escenarios COVID guardados      : {p2}")

    # 3. Predicciones de prueba
    errores = Y_test.values - test_pred.values
    df_prueba = pd.DataFrame({
        "Precio_Real"     : Y_test.values,
        "Precio_Predicho" : test_pred.values,
        "Error"           : errores,
    }, index=Y_test.index)
    df_prueba.index.name = "Fecha"
    p3 = os.path.join(ruta_salida, "prediccion_conjunto_prueba.csv")
    df_prueba.to_csv(p3, encoding="utf-8")
    rutas["prueba"] = p3
    print(f"  ✓ Predicciones prueba : {p3}")

    print(f"\n  RMSE del conjunto de prueba: {rmse_test:.4f}")

    return rutas


# =============================================================================
# SECCIÓN 9: PIPELINE PRINCIPAL (main)
# =============================================================================

def main() -> dict:
    """
    Pipeline completo de predicción SARIMAX del precio del cemento.

    Partición de datos: 70% Train / 15% Validación / 15% Test
    (misma partición que el modelo LSTM para comparación justa).

    Pasos:
      1. Cargar datos históricos del precio del cemento
      2. Cargar nivel del río completo e integrar con datos históricos
      3. Análisis exploratorio: tests de estacionariedad + 5 gráficos EDA
      4. División 70/15/15 (train / validación / test)
      5. Búsqueda automática de parámetros SARIMAX con auto_arima (sobre train)
      6. Modelo sobre TRAIN → predice VALIDACIÓN → RMSE validación
      7. Re-entrenamiento sobre TRAIN+VAL (85%) → predice TEST → RMSE test
      8. Modelo final (100% datos) → pronóstico futuro con IC 95%
      9. Pronóstico escenario COVID (Cuarentena=1)
     10. Generación de gráficos de resultados
     11. Exportación de CSVs
    """
    print("\n" + "═" * 70)
    print("  SARIMAX — PREDICCIÓN DEL PRECIO DEL CEMENTO")
    print("  Partición: 70% Train / 15% Validación / 15% Test")
    print("═" * 70)

    T_INICIO = time.time()

    # ── Paso 1-2: Carga de datos ─────────────────────────────────────────────
    df_cemento   = cargar_datos_historicos(RUTA_CEMENTO)
    df_nivel_rio = cargar_nivel_rio_completo(RUTA_NIVEL_RIO)

    # Unir nivel del río al dataset histórico
    print(f"{'━'*60}")
    print("  PASO 3 ▸ Integrando nivel del río con datos históricos")
    print(f"{'━'*60}")
    if "Nivel_Rio" in df_cemento.columns:
        df_cemento = df_cemento.drop(columns=["Nivel_Rio"])
    df_hist = df_cemento.join(df_nivel_rio[["Nivel_Rio"]], how="left")
    n_nan_nivel = df_hist["Nivel_Rio"].isnull().sum()
    if n_nan_nivel > 0:
        print(f"  ⚠  {n_nan_nivel} valores de Nivel_Rio sin correspondencia → interpolación.")
        df_hist["Nivel_Rio"] = df_hist["Nivel_Rio"].interpolate(method="linear").bfill().ffill()
    print(f"  ✓  Dataset histórico integrado: {len(df_hist)} observaciones con Nivel_Rio")

    # Exógenas futuras
    ultima_fecha = df_hist.index[-1]
    df_future_exog = df_nivel_rio[df_nivel_rio.index > ultima_fecha][["Nivel_Rio"]].copy()
    df_future_exog["Cuarentena"] = 0
    df_future_exog = df_future_exog[["Nivel_Rio", "Cuarentena"]]
    print(f"  ✓  Exógenas futuras: {len(df_future_exog)} meses "
          f"({df_future_exog.index[0].strftime('%Y-%m')} → "
          f"{df_future_exog.index[-1].strftime('%Y-%m')})")
    print()

    endogena_Y = df_hist["Precio_Cemento"]
    exogenas_X = df_hist[["Nivel_Rio", "Cuarentena"]]

    # ── Paso 3: Análisis exploratorio ────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 4 ▸ Análisis Exploratorio de Datos (EDA)")
    print(f"{'━'*60}")

    test_estacionariedad(endogena_Y, "Precio del Cemento")
    test_estacionariedad(df_hist["Nivel_Rio"], "Nivel del Río")

    grafico_01_series_temporales(df_hist, DIR_GRAFICOS)
    grafico_02_descomposicion(endogena_Y, PERIODO_ESTACIONAL, DIR_GRAFICOS)
    grafico_03_acf_pacf(endogena_Y, lags=40, dir_salida=DIR_GRAFICOS)
    grafico_04_correlacion(df_hist, DIR_GRAFICOS)
    grafico_05_patron_estacional(endogena_Y, DIR_GRAFICOS)

    # ── Paso 4: División 70 / 15 / 15 ───────────────────────────────────────
    # Se divide la serie temporal de forma cronológica (sin mezclar) en
    # tres subconjuntos contiguos, respetando el orden temporal:
    #
    #   |--- 70% Train ---|--- 15% Val ---|--- 15% Test ---|
    #   2012-01          2018-XX         2020-XX          2023-XX
    #
    # Es fundamental NO mezclar datos aleatoriamente (como en clasificación)
    # porque en series temporales el modelo aprende patrones temporales:
    # si se mezclan, se produce "data leakage" (el modelo vería datos futuros
    # durante el entrenamiento, inflando artificialmente las métricas).
    print(f"\n{'━'*60}")
    print("  PASO 5 ▸ División Train (70%) / Validación (15%) / Test (15%)")
    print(f"{'━'*60}")

    n_total = len(endogena_Y)
    n_train = int(n_total * RATIO_TRAIN)
    n_val   = int(n_total * RATIO_VAL)
    # n_test = lo que sobra para no perder observaciones por redondeo
    n_test  = n_total - n_train - n_val

    Y_train = endogena_Y.iloc[:n_train]
    Y_val   = endogena_Y.iloc[n_train:n_train + n_val]
    Y_test  = endogena_Y.iloc[n_train + n_val:]
    X_train = exogenas_X.iloc[:n_train]
    X_val   = exogenas_X.iloc[n_train:n_train + n_val]
    X_test  = exogenas_X.iloc[n_train + n_val:]

    print(f"  Total : {n_total} observaciones")
    print(f"  Train : {len(Y_train)} obs ({len(Y_train)/n_total*100:.0f}%)  "
          f"({Y_train.index[0].strftime('%Y-%m')} → {Y_train.index[-1].strftime('%Y-%m')})")
    print(f"  Val   : {len(Y_val)} obs  ({len(Y_val)/n_total*100:.0f}%)  "
          f"({Y_val.index[0].strftime('%Y-%m')} → {Y_val.index[-1].strftime('%Y-%m')})")
    print(f"  Test  : {len(Y_test)} obs  ({len(Y_test)/n_total*100:.0f}%)  "
          f"({Y_test.index[0].strftime('%Y-%m')} → {Y_test.index[-1].strftime('%Y-%m')})")

    # ── Paso 5: Búsqueda de parámetros (solo con TRAIN 70%) ────────────────
    # auto_arima usa SOLAMENTE el 70% de entrenamiento para seleccionar los
    # órdenes (p,d,q)(P,D,Q,m) óptimos mediante el criterio AIC.
    # Esto equivale al tuning de hiperparámetros del LSTM (learning rate,
    # número de capas, neuronas, etc.) que se hace sobre el train set.
    best_order, best_seasonal_order, _ = buscar_parametros_auto(
        Y_train, X_train, m=PERIODO_ESTACIONAL,
    )

    # ── Paso 6: Modelo sobre TRAIN → predice VALIDACIÓN ─────────────────────
    # Se entrena el modelo con el 70% (train) y se evalúa prediciendo
    # sobre el 15% de validación. El RMSE de validación obtenido aquí
    # es comparable al "validation loss" del LSTM.
    print(f"\n{'━'*60}")
    print("  PASO 6 ▸ Evaluación sobre conjunto de VALIDACIÓN")
    print(f"{'━'*60}")

    res_train = entrenar_sarimax(
        Y_train, X_train, best_order, best_seasonal_order,
        nombre=f"Modelo SARIMAX{best_order}x{best_seasonal_order} (train 70%)",
    )

    forecast_obj_val = res_train.get_forecast(steps=len(Y_val), exog=X_val)
    val_pred = forecast_obj_val.predicted_mean
    val_pred.index = Y_val.index

    rmse_val = calcular_rmse(Y_val, val_pred, "Conjunto de Validación (15%)")

    # ── Paso 7: Re-entrenar con TRAIN+VAL (85%) → predice TEST ──────────────
    # Se re-entrena el modelo usando el 85% de los datos (train + validación)
    # con los mismos parámetros (p,d,q)(P,D,Q,m) ya seleccionados.
    # Luego se predice sobre el 15% de test (datos nunca vistos).
    # El RMSE de test obtenido aquí es la métrica FINAL comparable con
    # el RMSE de test del LSTM.
    print(f"\n{'━'*60}")
    print("  PASO 7 ▸ Re-entrenamiento (train+val 85%) → Evaluación sobre TEST")
    print(f"{'━'*60}")

    Y_train_val = pd.concat([Y_train, Y_val])
    X_train_val = pd.concat([X_train, X_val])

    res_train_val = entrenar_sarimax(
        Y_train_val, X_train_val, best_order, best_seasonal_order,
        nombre=f"Modelo SARIMAX{best_order}x{best_seasonal_order} (train+val 85%)",
    )

    forecast_obj_test = res_train_val.get_forecast(steps=len(Y_test), exog=X_test)
    test_pred = forecast_obj_test.predicted_mean
    test_pred.index = Y_test.index

    rmse_test = calcular_rmse(Y_test, test_pred, "Conjunto de Test (15%)")

    # ── Paso 8: Modelo final (100% datos) → pronóstico futuro ───────────────
    # Finalmente, se entrena un último modelo usando el 100% de los datos
    # históricos (ya no hay datos reservados para evaluación) para generar
    # el pronóstico futuro con la mayor cantidad de información posible.
    print(f"\n{'━'*60}")
    print("  PASO 8 ▸ Modelo final (re-entrenado con el 100% de los datos)")
    print(f"{'━'*60}")

    final_res = entrenar_sarimax(
        endogena_Y, exogenas_X, best_order, best_seasonal_order,
        nombre=f"Modelo Final SARIMAX{best_order}x{best_seasonal_order} (100%)",
    )
    print(f"\n{final_res.summary()}")

    # ── Paso 9: Pronóstico futuro con IC 95% ─────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  PASO 9 ▸ Generando pronóstico de {len(df_future_exog)} meses")
    print(f"{'━'*60}")

    forecast_result = final_res.get_forecast(
        steps=len(df_future_exog), exog=df_future_exog,
    )
    forecast_media = forecast_result.predicted_mean
    forecast_ic    = forecast_result.conf_int(alpha=ALPHA_IC)

    forecast_media.index = df_future_exog.index
    forecast_ic.index    = df_future_exog.index

    forecast_df = pd.DataFrame({
        "Media"          : forecast_media.values,
        "IC_Inferior"    : forecast_ic.iloc[:, 0].values,
        "IC_Superior"    : forecast_ic.iloc[:, 1].values,
        "Nivel_Rio_Exog" : df_future_exog["Nivel_Rio"].values,
        "Cuarentena_Covid" : df_future_exog["Cuarentena"].values,
    }, index=df_future_exog.index)
    forecast_df.index.name = "Fecha"

    print("\n  Pronóstico futuro (Sin COVID — Cuarentena=0):")
    print(forecast_df.round(4).to_string())

    # ── Paso 10: Escenario alternativo Con COVID (Cuarentena = 1) ────────────
    df_future_exog_covid = df_future_exog.copy()
    df_future_exog_covid["Cuarentena"] = 1

    forecast_result_covid = final_res.get_forecast(
        steps=len(df_future_exog_covid), exog=df_future_exog_covid,
    )
    forecast_media_covid = forecast_result_covid.predicted_mean
    forecast_ic_covid    = forecast_result_covid.conf_int(alpha=ALPHA_IC)
    forecast_media_covid.index = df_future_exog_covid.index
    forecast_ic_covid.index    = df_future_exog_covid.index

    forecast_df_covid = pd.DataFrame({
        "Media"           : forecast_media_covid.values,
        "IC_Inferior"     : forecast_ic_covid.iloc[:, 0].values,
        "IC_Superior"     : forecast_ic_covid.iloc[:, 1].values,
        "Nivel_Rio_Exog"  : df_future_exog_covid["Nivel_Rio"].values,
        "Cuarentena_Covid": df_future_exog_covid["Cuarentena"].values,
    }, index=df_future_exog_covid.index)
    forecast_df_covid.index.name = "Fecha"

    print("\n  Pronóstico futuro (Con COVID — Cuarentena=1):")
    print(forecast_df_covid.round(4).to_string())

    # ── Paso 11: Gráficos de resultados ─────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 11 ▸ Generando gráficos de resultados")
    print(f"{'━'*60}")

    # Para el gráfico de pronóstico completo usamos train+val como "entrenamiento"
    # y test como "prueba", que es la evaluación final
    grafico_06_pronostico_completo(
        Y_train_val, Y_test, test_pred, forecast_df, rmse_test, DIR_GRAFICOS,
    )
    grafico_07_diagnostico_residuos(final_res, DIR_GRAFICOS)
    grafico_08_comparacion_prueba(Y_test, test_pred, rmse_test, DIR_GRAFICOS)
    grafico_09_escenarios_covid(forecast_df, forecast_df_covid, endogena_Y, DIR_GRAFICOS)

    # ── Paso 12: Exportar resultados ─────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 12 ▸ Exportando resultados")
    print(f"{'━'*60}")

    guardar_resultados_csv(
        forecast_df, forecast_df_covid, Y_test, test_pred, rmse_test, RUTA_SALIDA,
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    T_TOTAL = time.time() - T_INICIO
    print(f"\n{'═'*70}")
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'═'*70}")
    print(f"  Partición       : 70% Train / 15% Val / 15% Test")
    print(f"  Modelo          : SARIMAX{best_order}x{best_seasonal_order}")
    print(f"  RMSE (validación): {rmse_val:.4f}")
    print(f"  RMSE (test)      : {rmse_test:.4f}")
    print(f"  Tiempo total     : {T_TOTAL:.1f} s ({T_TOTAL/60:.1f} min)")
    print(f"  Gráficos en      : {DIR_GRAFICOS}")
    print(f"  Resultados en    : {RUTA_SALIDA}")
    print(f"{'═'*70}\n")

    return {
        "modelo"        : final_res,
        "forecast"      : forecast_df,
        "rmse_val"      : rmse_val,
        "rmse_test"     : rmse_test,
        "parametros"    : {
            "order"         : best_order,
            "seasonal_order": best_seasonal_order,
        },
    }


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    resultados = main()
