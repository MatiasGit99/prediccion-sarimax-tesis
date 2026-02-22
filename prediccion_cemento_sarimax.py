# =============================================================================
# prediccion_cemento_sarimax.py
# =============================================================================
# MODELO SARIMAX - PREDICCIÓN DEL PRECIO DEL CEMENTO
# Autor: Tesis de Grado / Posgrado
# Entorno: Kaggle (GPU/CPU)
# Descripción: Pipeline completo de series temporales con SARIMAX,
#              validación cruzada rolling, múltiples gráficos para tesis
#              e informes detallados en CSV y texto.
#
# Archivos de entrada requeridos (en /kaggle/input/dataset-sarimax-cemento/):
#   - precios_cemento_interpolado.csv   : Serie histórica del precio del cemento
#                                         más indicador binario de cuarentena COVID-19.
#   - nivel_rio_minimo_mensual.csv      : Nivel mínimo mensual del río, cubre tanto
#                                         el período histórico (para unirse al dataset
#                                         de cemento) como el período futuro (usado
#                                         como variable exógena en el pronóstico).
#
# Nota sobre selección de hiperparámetros SARIMAX:
#   A diferencia de modelos de ML (redes neuronales, XGBoost), SARIMAX NO usa
#   trials aleatorios/bayesianos. El espacio de búsqueda es discreto y pequeño:
#     p, q ∈ [0, 3]   (componentes AR y MA ordinarios)
#     P, Q ∈ [0, 2]   (componentes AR y MA estacionales)
#     d, D            (detectados automáticamente via tests ADF y OCSB)
#   auto_arima implementa el algoritmo stepwise de Hyndman-Khandakar: parte de
#   un modelo base y evalúa modelos vecinos (±1 en p/q/P/Q), moviéndose al
#   vecino con menor AIC hasta alcanzar un mínimo local. Típicamente evalúa
#   entre 15 y 30 modelos candidatos (visible con trace=True).
#   Rangos actuales: max_p=5, max_q=5, max_P=3, max_Q=3.
# =============================================================================

# =============================================================================
# SECCIÓN 0: INSTALACIÓN DE DEPENDENCIAS
# =============================================================================
# Nota: En Kaggle, la mayoría de estas librerías ya están instaladas.
# Se instalan solo las que podrían faltar.

import subprocess
import sys

def _instalar_si_falta(paquete: str, nombre_import: str | None = None) -> None:
    """Intenta importar un paquete; si falla, lo instala via pip."""
    nombre = nombre_import or paquete
    try:
        __import__(nombre)
    except ImportError:
        print(f"  Instalando {paquete}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", paquete, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

_instalar_si_falta("pmdarima")
_instalar_si_falta("openpyxl")


# =============================================================================
# SECCIÓN 1: IMPORTACIONES
# =============================================================================

import os
import time
import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")          # Backend sin pantalla (compatible con Kaggle)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from pmdarima import auto_arima

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
PERIODO_ESTACIONAL = 12   # Mensual → m = 12
N_TEST             = 15   # Meses reservados para evaluación final
ALPHA_IC           = 0.05 # Nivel de significancia para intervalos de confianza (95 %)
SEMILLA            = 42

# ── Paleta de colores para tesis (accesible para daltonismo) ─────────────────
COLORES = {
    "train"    : "#1565C0",   # Azul oscuro
    "test"     : "#E65100",   # Naranja oscuro
    "fit"      : "#2E7D32",   # Verde oscuro
    "forecast" : "#B71C1C",   # Rojo oscuro
    "ci_95"    : "#EF9A9A",   # Rojo claro (IC 95 %)
    "ci_80"    : "#FFCDD2",   # Rojo muy claro (IC 80 %)
    "nivel"    : "#6A1B9A",   # Violeta
    "covid"    : "#F44336",   # Rojo
    "neutral"  : "#546E7A",   # Gris azulado
    "pos_err"  : "#43A047",   # Verde (sobreestimación)
    "neg_err"  : "#E53935",   # Rojo (subestimación)
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
# SECCIÓN 3: VERIFICACIÓN DE HARDWARE (GPU)
# =============================================================================

def verificar_hardware() -> dict:
    """
    Verifica el hardware disponible en el entorno Kaggle.

    Nota sobre GPU y SARIMAX
    ─────────────────────────
    SARIMAX (statsmodels) es un algoritmo puramente CPU-bound; sus rutinas
    de optimización (L-BFGS, BFGS) no tienen implementación CUDA.
    Sin embargo, el acelerador GPU de Kaggle ofrece:
      • Mayor RAM (16 GB GPU + 30 GB RAM vs 30 GB RAM solo).
      • Acceso a cuML (RAPIDS) para preprocesamiento vectorizado en GPU.
      • Entorno con más capacidad de cómputo en paralelo para auto_arima
        cuando usa n_jobs=-1 (OpenMP en CPU).
    Si cuPy/cuML están disponibles, se usan para operaciones matriciales
    de preprocesamiento y cálculo de métricas.
    """
    info = {"gpu": False, "gpu_name": None, "ram_gb": None}

    # Detectar GPU con nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            info["gpu"] = True
            info["gpu_name"] = r.stdout.strip().split("\n")[0]
            print(f"  [GPU] Detectada: {info['gpu_name']}")
        else:
            print("  [HW]  GPU no detectada → usando CPU.")
    except Exception:
        print("  [HW]  nvidia-smi no disponible → usando CPU.")

    # Detectar RAM disponible
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        print(f"  [RAM] {info['ram_gb']} GB disponibles.")
    except ImportError:
        pass

    # Intentar usar cuPy para operaciones NumPy si hay GPU
    if info["gpu"]:
        try:
            import cupy as cp            # noqa: F401
            info["array_module"] = "cupy"
            print("  [GPU] cuPy disponible → aceleración GPU para NumPy ops.")
        except ImportError:
            info["array_module"] = "numpy"
            print("  [GPU] cuPy no disponible → NumPy (CPU) para operaciones matriciales.")
    else:
        info["array_module"] = "numpy"

    return info


# =============================================================================
# SECCIÓN 4: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def _limpiar_columna_numerica(serie: pd.Series) -> pd.Series:
    """Convierte una columna que puede tener comas decimales a float64."""
    return serie.astype(str).str.replace(",", ".", regex=False).astype(float)


def cargar_datos_historicos(ruta: str) -> pd.DataFrame:
    """
    Carga el CSV de precios de cemento interpolados.

    Columnas esperadas en el CSV (se aceptan variaciones de nombre):
      - Fecha               → índice temporal (DatetimeIndex, frecuencia 'MS')
      - Precio_Promedio_Polinomial_2 / Precio_Cemento → variable endógena
      - Nivel / Nivel_Rio   → nivel del río (variable exógena continua)
      - Cuarentena_Covid / Cuarentena → indicador binario COVID-19

    Transformaciones aplicadas:
      1. Renombrado de columnas para uniformidad.
      2. Conversión de comas decimales → puntos (formato europeo → float).
      3. Establecimiento de frecuencia mensual inicio-de-mes ('MS').
      4. Interpolación lineal para valores faltantes (si los hay).
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
    # Si ninguna coincidió exactamente, buscar la primera que contenga "precio"
    if precio_col is None:
        for c in df.columns:
            if "precio" in c.lower():
                precio_col = c
                break
    if precio_col and precio_col != "Precio_Cemento":
        rename[precio_col] = "Precio_Cemento"

    # Nivel y Cuarentena: buscar la primera coincidencia
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

    # Reporte
    print(f"  ✓  {len(df)} observaciones | "
          f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")
    print("\n  Estadísticas descriptivas:")
    print(df.describe().round(4).to_string(index=True))
    print()

    return df


def cargar_nivel_rio_completo(ruta: str) -> pd.DataFrame:
    """
    Carga el CSV completo del nivel mínimo mensual del río.

    Columnas esperadas:
      - Fecha  → índice temporal
      - Nivel  → nivel mínimo mensual del río

    Este archivo contiene tanto el período histórico (que se unirá a los
    datos del cemento vía join) como el período futuro (que se usará como
    variable exógena en el pronóstico). La separación entre ambos períodos
    se realiza en main() según la última fecha del dataset histórico de cemento.
    """
    print(f"\n{'━'*60}")
    print("  PASO 2 ▸ Cargando datos del nivel del río (histórico + futuro)")
    print(f"{'━'*60}")
    print(f"  Archivo: {ruta}")

    df = pd.read_csv(ruta, index_col="Fecha", parse_dates=True, decimal=",")

    # Renombrar columna de nivel
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
# SECCIÓN 5: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

def test_estacionariedad(serie: pd.Series, nombre: str = "Serie") -> dict:
    """
    Ejecuta los tests ADF y KPSS para evaluar estacionariedad.

    Test ADF (Augmented Dickey-Fuller):
      H₀: La serie tiene raíz unitaria (NO es estacionaria).
      Se rechaza H₀ si p-valor < 0.05 → serie estacionaria.

    Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
      H₀: La serie ES estacionaria alrededor de una tendencia determinista.
      Se rechaza H₀ si p-valor < 0.05 → serie NO estacionaria.

    Interpretación conjunta:
      ADF rechaza H₀ Y KPSS no rechaza H₀ → Serie estacionaria (confirmado).
      ADF no rechaza H₀ Y KPSS rechaza H₀ → Serie NO estacionaria (aplicar diferenciación).
      Resultados contradictorios → Posible tendencia estocástica (inspección visual).
    """
    res = {}
    s = serie.dropna()

    # ADF
    adf = adfuller(s, autolag="AIC")
    res["adf_stat"]        = adf[0]
    res["adf_pvalue"]      = adf[1]
    res["adf_estacionaria"] = adf[1] < 0.05

    # KPSS
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
    """
    Gráfico 1: Las tres series temporales apiladas verticalmente.
    Muestra precio del cemento, nivel del río e indicador de cuarentena.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        "Series Temporales: Precio del Cemento, Nivel del Río e Indicador COVID-19",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # Panel 1: Precio del cemento
    ax = axes[0]
    ax.plot(df.index, df["Precio_Cemento"], color=COLORES["train"], lw=1.8)
    ax.fill_between(df.index, df["Precio_Cemento"], alpha=0.08, color=COLORES["train"])
    ax.set_ylabel("Precio ($/bolsa)", fontsize=10)
    ax.set_title("Precio del Cemento", fontsize=11)

    # Panel 2: Nivel del río
    ax = axes[1]
    ax.plot(df.index, df["Nivel_Rio"], color=COLORES["nivel"], lw=1.8)
    ax.fill_between(df.index, df["Nivel_Rio"], alpha=0.08, color=COLORES["nivel"])
    ax.set_ylabel("Nivel (m)", fontsize=10)
    ax.set_title("Nivel del Río (Variable Exógena)", fontsize=11)

    # Panel 3: Cuarentena
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
    """
    Gráfico 2: Descomposición aditiva STL de la serie de precios.
    Separa la serie en: observada, tendencia, estacionalidad y residuo.
    """
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
    """
    Gráfico 3: ACF y PACF de la serie original y su primera diferencia.
    Herramienta diagnóstica para identificar los órdenes p, q del ARIMA.
    """
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
    """
    Gráfico 4: Matriz de correlación de Pearson y dispersión Precio vs Nivel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Análisis de Correlación entre Variables",
                 fontsize=13, fontweight="bold")

    # Mapa de calor de correlaciones
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
        vmin=-1, vmax=1, ax=axes[0], linewidths=0.6,
        cbar_kws={"label": "Correlación de Pearson", "shrink": 0.8},
        annot_kws={"size": 11},
    )
    axes[0].set_title("Matriz de Correlación de Pearson", fontsize=11)

    # Dispersión: Precio vs Nivel del Río (coloreado por tiempo)
    scatter = axes[1].scatter(
        df["Nivel_Rio"], df["Precio_Cemento"],
        c=np.arange(len(df)), cmap="plasma", alpha=0.75, s=45, zorder=5,
    )
    plt.colorbar(scatter, ax=axes[1], label="Índice temporal (meses)")

    # Línea de regresión
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
    """
    Gráfico 5: Boxplot por mes y evolución del precio promedio anual.
    Revela si existe una estacionalidad mensual sistemática en el precio.
    """
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

    # Boxplot mensual
    data_mes = [df_t.loc[df_t["Mes"] == m, "Precio"].values for m in range(1, 13)]
    bp = axes[0].boxplot(data_mes, labels=meses_labels, patch_artist=True,
                          notch=False, medianprops={"color": "red", "lw": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
        patch.set_alpha(0.7)
    axes[0].set_title("Distribución del Precio por Mes del Año", fontsize=11)
    axes[0].set_xlabel("Mes", fontsize=10)
    axes[0].set_ylabel("Precio del Cemento", fontsize=10)

    # Precio promedio anual
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
# SECCIÓN 6: MODELADO SARIMAX
# =============================================================================

def buscar_parametros_auto(
    Y_train: pd.Series,
    X_train: pd.DataFrame,
    m: int = 12,
) -> tuple[tuple, tuple, object]:
    """
    Busca los órdenes ARIMA y estacionales óptimos usando pmdarima.auto_arima.

    Mecanismo de selección (stepwise de Hyndman-Khandakar):
    ──────────────────────────────────────────────────────────
    SARIMAX NO utiliza trials aleatorios ni optimización bayesiana.
    El espacio de hiperparámetros es discreto y acotado:

      p, q ∈ [0, 5]   → componente autorregresivo y media móvil ordinarios
      d               → detectado automáticamente via test ADF
      P, Q ∈ [0, 3]   → componentes AR y MA estacionales
      D               → detectado automáticamente via test OCSB
      m = 12          → período estacional (mensual)

    El algoritmo:
      1. Determina d y D mediante tests estadísticos (no por búsqueda).
      2. Parte de un modelo base (típicamente ARIMA(2,d,2)(1,D,1)₁₂).
      3. Evalúa modelos vecinos (variando p, q, P, Q en ±1).
      4. Se desplaza al vecino con menor AIC.
      5. Repite hasta que ningún vecino mejora → mínimo local.

    Con stepwise=True se evalúan típicamente entre 15 y 30 modelos
    (visible en la salida de trace=True), frente a cientos en búsqueda
    exhaustiva (stepwise=False), con resultados equivalentes en la práctica.

    Nota sobre el rango de búsqueda (max_p=5, max_q=5, max_P=3, max_Q=3):
      Ampliar el rango respecto a valores anteriores (3/3/2/2) hace elegibles
      modelos de mayor orden sin forzarlos: AIC penaliza la complejidad y el
      algoritmo solo los seleccionará si mejoran genuinamente el ajuste neto.
      El límite de 5/3 es razonable para ~139 observaciones mensuales
      (regla orientativa: ≥ 10 observaciones por parámetro estimado).
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
        start_p        = 0, max_p = 5,  # Ampliado de 3 → 5
        start_q        = 0, max_q = 5,  # Ampliado de 3 → 5
        d              = None,          # Detección automática via test ADF
        start_P        = 0, max_P = 3,  # Ampliado de 2 → 3
        start_Q        = 0, max_Q = 3,  # Ampliado de 2 → 3
        D              = None,          # Detección automática via test OCSB
        seasonal       = True,
        m              = m,
        stepwise       = True,
        information_criterion = "aic",
        trace          = True,
        suppress_warnings = True,
        error_action   = "ignore",
        n_jobs         = -1,           # Todos los núcleos disponibles
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

    El modelo SARIMAX generaliza el SARIMA incorporando un vector de
    regresores externos X_t:

        Φ(B^m) φ(B) ∇^D ∇^d Yₜ = Θ(B^m) θ(B) εₜ + β Xₜ

    donde:
      • φ(B)  = 1 - φ₁B - … - φₚBᵖ         (parte AR ordinaria)
      • θ(B)  = 1 + θ₁B + … + θ_qB^q        (parte MA ordinaria)
      • Φ(B^m)= 1 - Φ₁B^m - … - Φ_PB^(Pm)  (parte AR estacional)
      • Θ(B^m)= 1 + Θ₁B^m + … + Θ_QB^(Qm)  (parte MA estacional)
      • ∇^d   = (1-B)^d                      (diferenciación ordinaria)
      • ∇^D   = (1-B^m)^D                    (diferenciación estacional)
      • β Xₜ  = regresión lineal de exógenas

    Parámetros de optimización:
      - method='lbfgs': Limited-memory BFGS, robusto para series largas.
      - maxiter=300   : Suficiente para convergencia en casi todos los casos.
      - enforce_stationarity=False : Permite soluciones cerca de la frontera.
      - enforce_invertibility=False: Ídem para la parte MA.
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
# SECCIÓN 7: MÉTRICAS DE EVALUACIÓN
# =============================================================================

def calcular_metricas(
    y_real: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    nombre: str = "",
) -> dict:
    """
    Calcula un conjunto completo de métricas de error de pronóstico.

    Métricas implementadas:
    ┌──────┬───────────────────────────────────┬────────────────────────────────┐
    │ RMSE │ Raíz del Error Cuadrático Medio    │ Penaliza errores grandes       │
    │ MAE  │ Error Absoluto Medio               │ Robusta a outliers             │
    │ MAPE │ Error Porcentual Absoluto Medio    │ Interpretación porcentual      │
    │ SMAPE│ MAPE Simétrico                     │ Evita asimetrías del MAPE      │
    │ R²   │ Coeficiente de Determinación       │ Proporción de varianza explicada│
    │ MaxAE│ Error Absoluto Máximo              │ Peor caso puntual              │
    └──────┴───────────────────────────────────┴────────────────────────────────┘
    """
    r = np.asarray(y_real, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = r != 0

    metricas = {
        "RMSE" : float(np.sqrt(mean_squared_error(r, p))),
        "MAE"  : float(mean_absolute_error(r, p)),
        "MAPE" : float(np.mean(np.abs((r[mask] - p[mask]) / r[mask])) * 100),
        "SMAPE": float(np.mean(2 * np.abs(r - p) / (np.abs(r) + np.abs(p))) * 100),
        "R2"   : float(r2_score(r, p)),
        "MaxAE": float(np.max(np.abs(r - p))),
    }

    if nombre:
        sep = "─" * 52
        print(f"\n  {sep}")
        print(f"  MÉTRICAS — {nombre}")
        print(f"  {sep}")
        print(f"  RMSE  (Raíz Error Cuadrático Medio) : {metricas['RMSE']:>10.4f}")
        print(f"  MAE   (Error Absoluto Medio)         : {metricas['MAE']:>10.4f}")
        print(f"  MAPE  (Error Porcentual Medio)       : {metricas['MAPE']:>9.2f} %")
        print(f"  SMAPE (Error Porcentual Simétrico)   : {metricas['SMAPE']:>9.2f} %")
        print(f"  R²    (Coeficiente Determinación)    : {metricas['R2']:>10.4f}")
        print(f"  MaxAE (Error Absoluto Máximo)        : {metricas['MaxAE']:>10.4f}")
        print(f"  {sep}")

    return metricas


# =============================================================================
# SECCIÓN 8: VALIDACIÓN CRUZADA ROLLING
# =============================================================================

def validacion_rolling(
    Y: pd.Series,
    X: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
    n_initial: int,
    step: int = 1,
) -> pd.DataFrame:
    """
    Validación cruzada con ventana expandible (expanding window CV).

    Metodología:
      1. Se entrena el modelo con las primeras n_initial observaciones.
      2. Se predice el siguiente paso (o `step` pasos).
      3. La ventana se expande en `step` observaciones.
      4. Se repite hasta agotar la serie.

    Esta estrategia respeta el orden temporal y proporciona una estimación
    más robusta del error de generalización que una única partición train/test.
    """
    print(f"\n{'━'*60}")
    print("  PASO 8 ▸ Validación cruzada con Rolling Forecast")
    print(f"{'━'*60}")
    print(f"  Ventana inicial : {n_initial} | Paso : {step}")

    reales, predichos, fechas = [], [], []
    n = len(Y)
    n_iter = 0

    for i in range(n_initial, n, step):
        try:
            mod = SARIMAX(
                Y.iloc[:i], exog=X.iloc[:i],
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            res = mod.fit(disp=False, method="lbfgs", maxiter=100)

            end_idx = min(i + step, n)
            pred = res.forecast(steps=end_idx - i, exog=X.iloc[i:end_idx])

            for j, val in enumerate(pred):
                idx = i + j
                if idx < n:
                    predichos.append(float(val))
                    reales.append(float(Y.iloc[idx]))
                    fechas.append(Y.index[idx])

            n_iter += 1
            if n_iter % 10 == 0:
                print(f"    Iteración {n_iter:3d} completada (fecha: {Y.index[i].strftime('%Y-%m')})")

        except Exception as e:
            print(f"    ⚠ Error en i={i}: {e}")
            continue

    df_roll = pd.DataFrame(
        {"Real": reales, "Predicho_Rolling": predichos},
        index=pd.DatetimeIndex(fechas),
    )

    if len(df_roll) > 0:
        calcular_metricas(df_roll["Real"], df_roll["Predicho_Rolling"],
                          f"Rolling Forecast CV ({n_iter} iteraciones)")
    else:
        print("  ⚠ Rolling forecast sin resultados.")

    return df_roll


# =============================================================================
# SECCIÓN 9: GRÁFICOS DE RESULTADOS DEL MODELO
# =============================================================================

def grafico_06_pronostico_completo(
    Y_train: pd.Series,
    Y_test: pd.Series,
    test_pred: pd.Series,
    forecast_df: pd.DataFrame,
    metricas: dict,
    dir_salida: str,
) -> str:
    """
    Gráfico 6: Vista completa del pronóstico — datos de entrenamiento,
    conjunto de prueba (real vs predicho) y pronóstico futuro con IC 95%.
    Panel superior: serie completa. Panel inferior: zoom sobre el período crítico.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 13))
    fig.suptitle(
        "Pronóstico del Precio del Cemento — Modelo SARIMAX\n"
        f"Parámetros: RMSE={metricas['RMSE']:.2f}  MAPE={metricas['MAPE']:.1f}%  R²={metricas['R2']:.3f}",
        fontsize=13, fontweight="bold",
    )

    for ax_idx, ax in enumerate(axes):
        # Datos de entrenamiento
        ax.plot(Y_train.index, Y_train.values,
                color=COLORES["train"], lw=1.5, label="Entrenamiento", zorder=3)

        # Datos reales del período de prueba
        ax.plot(Y_test.index, Y_test.values,
                color=COLORES["test"], lw=2.2, marker="o", ms=4,
                label="Real (prueba)", zorder=4)

        # Predicción sobre el período de prueba
        ax.plot(test_pred.index, test_pred.values,
                color=COLORES["fit"], lw=1.8, ls="--",
                label=f"Predicción prueba (RMSE={metricas['RMSE']:.2f})", zorder=4)

        # Pronóstico futuro + IC 95%
        ax.plot(forecast_df.index, forecast_df["Media"],
                color=COLORES["forecast"], lw=2.2, ls="--",
                marker="s", ms=3, label="Pronóstico futuro", zorder=5)
        ax.fill_between(
            forecast_df.index,
            forecast_df["IC_Inferior"], forecast_df["IC_Superior"],
            alpha=0.20, color=COLORES["forecast"], label="IC 95%",
        )

        # Líneas verticales divisoras
        ax.axvline(Y_test.index[0],    color="gray",  ls=":", lw=1.2, alpha=0.8)
        ax.axvline(forecast_df.index[0], color="black", ls=":", lw=1.2, alpha=0.8)

        ax.set_ylabel("Precio del Cemento", fontsize=10)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        if ax_idx == 0:
            ax.set_title("Vista Completa de la Serie", fontsize=11)
        else:
            # Zoom: últimos 48 meses de train + test + futuro
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
    Un buen modelo SARIMAX produce residuos que deben ser:
      1. Estacionarios (sin estructura temporal).
      2. Normalmente distribuidos (histograma y Q-Q plot).
      3. No autocorrelacionados (ACF de residuos dentro de bandas).
      4. Homocedásticos (residuos estandarizados sin patrón de embudo).
    """
    resid = resultado_modelo.resid

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("Diagnóstico Completo de Residuos — Modelo SARIMAX Final",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Residuos en el tiempo ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(resid.index, resid.values, color=COLORES["neutral"], lw=0.9)
    ax1.axhline(0, color="red", ls="--", lw=1.2, alpha=0.7)
    ax1.fill_between(resid.index, resid.values, 0, alpha=0.15, color=COLORES["neutral"])
    ax1.set_title("Residuos en el Tiempo (deben fluctuar aleatoriamente alrededor de 0)", fontsize=11)
    ax1.set_ylabel("Residuo", fontsize=10)

    # ── Panel 2: Histograma + curva normal teórica ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    mu_r, sigma_r = resid.mean(), resid.std()
    ax2.hist(resid.dropna(), bins=25, density=True,
             color="#42A5F5", alpha=0.70, edgecolor="white", label="Distribución real")
    x_n = np.linspace(resid.min(), resid.max(), 300)
    ax2.plot(x_n, stats.norm.pdf(x_n, mu_r, sigma_r), "r-", lw=2, label="Normal teórica")
    ax2.set_title("Distribución de Residuos vs Normal Teórica", fontsize=11)
    ax2.set_xlabel("Residuo", fontsize=10)
    ax2.set_ylabel("Densidad", fontsize=10)
    ax2.legend(fontsize=9)

    # ── Panel 3: Q-Q plot ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, r_qq) = stats.probplot(resid.dropna(), dist="norm")
    ax3.scatter(osm, osr, color="#42A5F5", alpha=0.6, s=22, zorder=5)
    ax3.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2, label=f"R²={r_qq**2:.4f}")
    ax3.set_title("Q-Q Plot de Residuos (normalidad)", fontsize=11)
    ax3.set_xlabel("Cuantiles teóricos (Normal)", fontsize=10)
    ax3.set_ylabel("Cuantiles observados", fontsize=10)
    ax3.legend(fontsize=9)

    # ── Panel 4: ACF de residuos ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    plot_acf(resid.dropna(), lags=30, ax=ax4, color="#42A5F5",
             title="ACF de Residuos (sin estructura → buen ajuste)")

    # ── Panel 5: Residuos estandarizados vs valores ajustados ───────────────
    ax5 = fig.add_subplot(gs[2, 1])
    resid_std = (resid - mu_r) / sigma_r
    fitted = resultado_modelo.fittedvalues
    ax5.scatter(fitted, resid_std, alpha=0.5, color="#42A5F5", s=22, zorder=5)
    ax5.axhline(0, color="red", ls="--", lw=1.2)
    ax5.axhline(2,  color="orange", ls=":", lw=1.2, alpha=0.7, label="±2σ (95%)")
    ax5.axhline(-2, color="orange", ls=":", lw=1.2, alpha=0.7)
    ax5.axhline(3,  color="red",    ls=":", lw=1.0, alpha=0.5, label="±3σ (99.7%)")
    ax5.axhline(-3, color="red",    ls=":", lw=1.0, alpha=0.5)
    ax5.set_title("Residuos Estandarizados vs Valores Ajustados (homocedasticidad)", fontsize=11)
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
    metricas: dict,
    dir_salida: str,
) -> str:
    """
    Gráfico 8: Análisis detallado del conjunto de prueba (4 paneles).
    Compara valores reales vs predichos, errores absolutos y porcentuales.
    """
    errores = Y_test.values - test_pred.values
    mape_mes = np.abs(errores / Y_test.values) * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Análisis del Conjunto de Prueba — Real vs Predicho\n"
        f"RMSE={metricas['RMSE']:.2f}  MAE={metricas['MAE']:.2f}  "
        f"MAPE={metricas['MAPE']:.1f}%  R²={metricas['R2']:.3f}",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: Líneas real vs predicho
    ax = axes[0, 0]
    ax.plot(Y_test.index, Y_test.values, "o-", color=COLORES["test"],
            lw=2, ms=5, label="Real")
    ax.plot(test_pred.index, test_pred.values, "s--", color=COLORES["fit"],
            lw=2, ms=5, label="Predicho")
    ax.set_title("Real vs Predicho en el Período de Prueba", fontsize=11)
    ax.set_ylabel("Precio del Cemento", fontsize=10)
    ax.legend(fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Panel 2: Scatter real vs predicho (linea de 45°)
    ax2 = axes[0, 1]
    ax2.scatter(Y_test.values, test_pred.values, color="#42A5F5", alpha=0.8, s=60, zorder=5)
    lim_min = min(Y_test.min(), test_pred.min()) * 0.98
    lim_max = max(Y_test.max(), test_pred.max()) * 1.02
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2, label="Predicción perfecta")
    ax2.set_xlim(lim_min, lim_max)
    ax2.set_ylim(lim_min, lim_max)
    ax2.set_title("Dispersión: Real vs Predicho", fontsize=11)
    ax2.set_xlabel("Precio Real", fontsize=10)
    ax2.set_ylabel("Precio Predicho", fontsize=10)
    ax2.legend(fontsize=9)

    # Panel 3: Error absoluto por período
    ax3 = axes[1, 0]
    colores_bar = [COLORES["pos_err"] if e >= 0 else COLORES["neg_err"] for e in errores]
    ax3.bar(range(len(errores)), errores, color=colores_bar, alpha=0.75)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.axhline(errores.mean(), color="blue", ls="--", lw=1.5,
                label=f"Error medio: {errores.mean():.2f}")
    ax3.set_title("Error Absoluto por Período (Real − Predicho)", fontsize=11)
    ax3.set_xlabel("Mes del período de prueba", fontsize=10)
    ax3.set_ylabel("Error", fontsize=10)
    ax3.legend(fontsize=9)
    leyenda_err = [Patch(fc=COLORES["pos_err"], alpha=0.75, label="Sobreestimación"),
                   Patch(fc=COLORES["neg_err"], alpha=0.75, label="Subestimación")]
    ax3.legend(handles=leyenda_err + ax3.get_legend_handles_labels()[0], fontsize=9)

    # Panel 4: MAPE por período
    ax4 = axes[1, 1]
    ax4.bar(range(len(mape_mes)), mape_mes, color="#7B1FA2", alpha=0.70)
    ax4.axhline(mape_mes.mean(), color="red", ls="--", lw=1.5,
                label=f"MAPE medio: {mape_mes.mean():.2f}%")
    ax4.set_title("Error Porcentual Absoluto por Período (MAPE%)", fontsize=11)
    ax4.set_xlabel("Mes del período de prueba", fontsize=10)
    ax4.set_ylabel("Error (%)", fontsize=10)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "08_comparacion_real_predicho.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_09_pronostico_ic(
    forecast_df: pd.DataFrame,
    Y_hist: pd.Series,
    dir_salida: str,
) -> str:
    """
    Gráfico 9: Pronóstico futuro con bandas de incertidumbre (IC 80% y 95%).
    Las bandas se amplían a medida que el horizonte temporal aumenta,
    reflejando la acumulación de incertidumbre en el pronóstico multi-paso.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Pronóstico Futuro del Precio del Cemento con Intervalos de Confianza\n"
        "Modelo SARIMAX — Horizonte de pronóstico: "
        f"{len(forecast_df)} meses",
        fontsize=12, fontweight="bold",
    )

    # IC 80% aproximado (± 1.28σ), usando interpolación entre media e IC 95%
    ic_rango = forecast_df["IC_Superior"] - forecast_df["IC_Inferior"]
    ic80_sup = forecast_df["Media"] + ic_rango * (1.28 / 1.96) / 2
    ic80_inf = forecast_df["Media"] - ic_rango * (1.28 / 1.96) / 2

    # ── Panel izquierdo: Histórico reciente + pronóstico ────────────────────
    ax = axes[0]
    n_hist = min(48, len(Y_hist))
    Y_rec = Y_hist.iloc[-n_hist:]
    ax.plot(Y_rec.index, Y_rec.values, color=COLORES["train"], lw=2,
            label=f"Histórico (últimos {n_hist} meses)")
    ax.plot(forecast_df.index, forecast_df["Media"], color=COLORES["forecast"],
            lw=2.5, ls="--", marker="o", ms=4, label="Pronóstico (media)")
    ax.fill_between(forecast_df.index,
                    forecast_df["IC_Inferior"], forecast_df["IC_Superior"],
                    alpha=0.18, color=COLORES["forecast"], label="IC 95%")
    ax.fill_between(forecast_df.index, ic80_inf, ic80_sup,
                    alpha=0.30, color=COLORES["forecast"], label="IC 80%")
    ax.axvline(forecast_df.index[0], color="gray", ls=":", lw=1.2, alpha=0.7)
    ax.set_title("Contexto Histórico + Pronóstico Futuro", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel("Precio del Cemento", fontsize=10)
    ax.legend(fontsize=9)

    # ── Panel derecho: Solo pronóstico con etiquetas de valor ───────────────
    ax2 = axes[1]
    ax2.plot(forecast_df.index, forecast_df["Media"], "o-",
             color=COLORES["forecast"], lw=2.2, ms=7, label="Pronóstico")
    ax2.fill_between(forecast_df.index,
                     forecast_df["IC_Inferior"], forecast_df["IC_Superior"],
                     alpha=0.20, color=COLORES["forecast"], label="IC 95%")
    ax2.fill_between(forecast_df.index, ic80_inf, ic80_sup,
                     alpha=0.35, color=COLORES["forecast"], label="IC 80%")

    # Etiquetas en valores alternos
    for i, (idx, row) in enumerate(forecast_df.iterrows()):
        if i % 2 == 0:
            ax2.annotate(
                f"{row['Media']:.1f}",
                xy=(idx, row["Media"]),
                xytext=(0, 10), textcoords="offset points",
                fontsize=8, ha="center", color="#B71C1C", fontweight="bold",
            )

    ax2.set_title("Detalle del Pronóstico Futuro con Valores", fontsize=11)
    ax2.set_xlabel("Fecha", fontsize=10)
    ax2.set_ylabel("Precio del Cemento", fontsize=10)
    ax2.legend(fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "09_pronostico_intervalo_confianza.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_10_rolling_forecast(
    df_rolling: pd.DataFrame,
    Y_hist: pd.Series,
    dir_salida: str,
) -> str:
    """
    Gráfico 10: Resultado de la validación cruzada rolling.
    Muestra cuán bien el modelo predice a un paso adelante de forma iterativa.
    """
    if len(df_rolling) == 0:
        print("  ⚠ Sin datos de rolling para graficar.")
        return ""

    errores_roll = df_rolling["Real"] - df_rolling["Predicho_Rolling"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Validación Cruzada con Rolling Forecast (Ventana Expandible)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(Y_hist.index, Y_hist.values, color="#B0BEC5", lw=1, alpha=0.4,
            label="Serie Completa")
    ax.plot(df_rolling.index, df_rolling["Real"], color=COLORES["test"],
            lw=1.8, label="Real (ventana rolling)")
    ax.plot(df_rolling.index, df_rolling["Predicho_Rolling"], "--",
            color=COLORES["fit"], lw=1.8, label="Predicho (1 paso adelante)")
    ax.set_title("Real vs Predicho — Rolling One-Step-Ahead Forecast", fontsize=11)
    ax.set_ylabel("Precio del Cemento", fontsize=10)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(df_rolling.index, errores_roll, color=COLORES["neutral"], lw=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.fill_between(df_rolling.index, errores_roll, 0,
                     where=(errores_roll >= 0), alpha=0.3,
                     color=COLORES["pos_err"], label="Sobreestimación")
    ax2.fill_between(df_rolling.index, errores_roll, 0,
                     where=(errores_roll < 0), alpha=0.3,
                     color=COLORES["neg_err"], label="Subestimación")
    ax2.set_title("Error del Rolling Forecast", fontsize=11)
    ax2.set_xlabel("Fecha", fontsize=10)
    ax2.set_ylabel("Error (Real − Predicho)", fontsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "10_rolling_forecast_cv.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


def grafico_11_ajuste_vs_real(
    Y_hist: pd.Series,
    resultado_modelo,
    dir_salida: str,
) -> str:
    """
    Gráfico 11: Valores ajustados del modelo final vs serie histórica completa.
    Permite evaluar el ajuste in-sample del modelo.
    """
    fitted = resultado_modelo.fittedvalues

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Ajuste In-Sample del Modelo SARIMAX Final",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(Y_hist.index, Y_hist.values, color=COLORES["train"],
            lw=1.5, label="Serie Histórica Real", alpha=0.9)
    ax.plot(fitted.index, fitted.values, color=COLORES["fit"],
            lw=1.2, ls="--", label="Valores Ajustados (in-sample)", alpha=0.85)
    ax.set_title("Serie Histórica vs Valores Ajustados", fontsize=11)
    ax.set_ylabel("Precio del Cemento", fontsize=10)
    ax.legend(fontsize=9)

    # Error in-sample
    err_insp = Y_hist - fitted
    ax2 = axes[1]
    ax2.fill_between(err_insp.index, err_insp.values, 0,
                     where=(err_insp >= 0), alpha=0.4,
                     color=COLORES["pos_err"], label="Sobre-ajuste")
    ax2.fill_between(err_insp.index, err_insp.values, 0,
                     where=(err_insp < 0), alpha=0.4,
                     color=COLORES["neg_err"], label="Sub-ajuste")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Error de Ajuste In-Sample (Real − Ajustado)", fontsize=11)
    ax2.set_xlabel("Fecha", fontsize=10)
    ax2.set_ylabel("Error", fontsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(dir_salida, "11_ajuste_insample.png")
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado: {ruta}")
    return ruta


# =============================================================================
# SECCIÓN 10: EXPORTACIÓN DE RESULTADOS
# =============================================================================

def guardar_resultados_csv(
    forecast_df: pd.DataFrame,
    Y_test: pd.Series,
    test_pred: pd.Series,
    metricas: dict,
    ruta_salida: str,
) -> dict:
    """
    Guarda tres archivos CSV en /kaggle/working/:

    1. pronostico_futuro_cemento.csv
       → Pronóstico mensual futuro con media e intervalos de confianza.

    2. prediccion_conjunto_prueba.csv
       → Comparación real vs predicho para los N_TEST meses de evaluación.

    3. metricas_modelo.csv
       → Tabla resumen con todas las métricas de evaluación.
    """
    rutas = {}

    # 1. Pronóstico futuro
    p1 = os.path.join(ruta_salida, "pronostico_futuro_cemento.csv")
    forecast_df.to_csv(p1, encoding="utf-8")
    rutas["pronostico"] = p1
    print(f"  ✓ Pronóstico guardado : {p1}")

    # 2. Predicciones de prueba con errores
    errores = Y_test.values - test_pred.values
    df_prueba = pd.DataFrame({
        "Precio_Real"           : Y_test.values,
        "Precio_Predicho"       : test_pred.values,
        "Error_Absoluto"        : errores,
        "Error_Porcentual_pct"  : np.abs(errores / Y_test.values) * 100,
        "Error_Relativo_pct"    : errores / Y_test.values * 100,
    }, index=Y_test.index)
    df_prueba.index.name = "Fecha"

    p2 = os.path.join(ruta_salida, "prediccion_conjunto_prueba.csv")
    df_prueba.to_csv(p2, encoding="utf-8")
    rutas["prueba"] = p2
    print(f"  ✓ Predicciones prueba : {p2}")

    # 3. Métricas
    df_met = pd.DataFrame([metricas])
    p3 = os.path.join(ruta_salida, "metricas_modelo.csv")
    df_met.to_csv(p3, index=False, encoding="utf-8")
    rutas["metricas"] = p3
    print(f"  ✓ Métricas guardadas  : {p3}")

    return rutas


def generar_informe_texto(
    metricas_test: dict,
    metricas_rolling: dict,
    best_order: tuple,
    best_seasonal_order: tuple,
    resultado_final,
    df_hist: pd.DataFrame,
    forecast_df: pd.DataFrame,
    hw_info: dict,
    ruta_salida: str,
) -> str:
    """
    Genera un informe completo en texto plano (informe_sarimax.txt)
    con: resumen del dataset, parámetros del modelo, métricas de evaluación,
    tests estadísticos, pronóstico y resumen del modelo estadístico.
    """
    # Test de Ljung-Box (H0: no hay autocorrelación en los residuos)
    try:
        ljung = acorr_ljungbox(resultado_final.resid, lags=[10, 20], return_df=True)
        ljung_str = ljung.to_string()
    except Exception:
        ljung_str = "No disponible"

    # Test de normalidad de Shapiro-Wilk sobre residuos
    resid_vals = resultado_final.resid.dropna().values
    try:
        sw_stat, sw_p = stats.shapiro(resid_vals[:min(5000, len(resid_vals))])
        sw_str = f"W={sw_stat:.4f}  p-valor={sw_p:.4f}  → {'Normal ✓' if sw_p > 0.05 else 'No Normal ✗'}"
    except Exception:
        sw_str = "No disponible"

    def _fmt(val, fmt=".4f", suf=""):
        try:
            return f"{val:{fmt}}{suf}"
        except Exception:
            return "N/A"

    informe = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         INFORME DE MODELO SARIMAX — PREDICCIÓN PRECIO DEL CEMENTO          ║
╚══════════════════════════════════════════════════════════════════════════════╝
  Fecha de generación : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Hardware            : {'GPU: ' + str(hw_info.get('gpu_name','N/A')) if hw_info.get('gpu') else 'CPU'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. RESUMEN DEL DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Observaciones totales     : {len(df_hist)}
  Período                   : {df_hist.index[0].strftime('%Y-%m')} → {df_hist.index[-1].strftime('%Y-%m')}
  Variable endógena         : Precio_Cemento
  Variables exógenas        : Nivel_Rio, Cuarentena
  Precio mínimo             : {_fmt(df_hist['Precio_Cemento'].min())}
  Precio máximo             : {_fmt(df_hist['Precio_Cemento'].max())}
  Precio promedio           : {_fmt(df_hist['Precio_Cemento'].mean())}
  Desviación estándar       : {_fmt(df_hist['Precio_Cemento'].std())}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2. PARÁMETROS DEL MODELO SARIMAX SELECCIONADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Orden ARIMA (p,d,q)          : {best_order}
  Orden Estacional (P,D,Q,m)   : {best_seasonal_order}
  AIC                          : {_fmt(resultado_final.aic)}
  BIC                          : {_fmt(resultado_final.bic)}
  Log-Likelihood               : {_fmt(resultado_final.llf)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3. MÉTRICAS DE EVALUACIÓN — CONJUNTO DE PRUEBA ({N_TEST} meses)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RMSE  (Raíz Error Cuadrático Medio)   : {_fmt(metricas_test.get('RMSE'))}
  MAE   (Error Absoluto Medio)           : {_fmt(metricas_test.get('MAE'))}
  MAPE  (Error Porcentual Medio)         : {_fmt(metricas_test.get('MAPE'), '.2f', ' %')}
  SMAPE (Error Porcentual Simétrico)     : {_fmt(metricas_test.get('SMAPE'), '.2f', ' %')}
  R²    (Coeficiente de Determinación)   : {_fmt(metricas_test.get('R2'))}
  MaxAE (Error Absoluto Máximo)          : {_fmt(metricas_test.get('MaxAE'))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4. MÉTRICAS DE VALIDACIÓN CRUZADA — ROLLING FORECAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RMSE  : {_fmt(metricas_rolling.get('RMSE'))}
  MAE   : {_fmt(metricas_rolling.get('MAE'))}
  MAPE  : {_fmt(metricas_rolling.get('MAPE'), '.2f', ' %')}
  R²    : {_fmt(metricas_rolling.get('R2'))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5. TESTS ESTADÍSTICOS SOBRE RESIDUOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Ljung-Box (H₀: residuos son ruido blanco):
{ljung_str}

  Shapiro-Wilk (H₀: residuos siguen distribución normal):
    {sw_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  6. PRONÓSTICO FUTURO (con Intervalos de Confianza al 95%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{forecast_df.round(4).to_string()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  7. RESUMEN ESTADÍSTICO COMPLETO DEL MODELO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{resultado_final.summary().as_text()}

══════════════════════════════════════════════════════════════════════════════════
"""

    ruta = os.path.join(ruta_salida, "informe_sarimax.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(informe)
    print(f"  ✓ Informe guardado    : {ruta}")
    return ruta


# =============================================================================
# SECCIÓN 11: PIPELINE PRINCIPAL (main)
# =============================================================================

def main() -> dict:
    """
    Pipeline completo de predicción SARIMAX del precio del cemento.

    Pasos:
      1. Verificar hardware disponible (GPU/CPU)
      2. Cargar datos históricos del precio del cemento (precios_cemento_interpolado.csv)
      3. Cargar nivel del río completo (nivel_rio_minimo_mensual.csv):
           3a. Unir el tramo histórico al dataset de cemento mediante join temporal.
           3b. Separar automáticamente las fechas posteriores al último dato
               histórico como variables exógenas futuras para el pronóstico.
      4. Análisis exploratorio: tests de estacionariedad + 5 gráficos EDA
      5. División train/test (N_TEST meses para evaluación)
      6. Búsqueda automática de parámetros SARIMAX con auto_arima
           Algoritmo stepwise de Hyndman-Khandakar: evalúa modelos vecinos
           (±1 en p/q/P/Q) minimizando AIC; d y D se detectan por tests
           estadísticos (ADF y OCSB). No usa trials aleatorios.
      7. Entrenamiento del modelo de validación (sobre TRAIN) + métricas TEST
      8. Entrenamiento del modelo final (sobre TODOS los datos)
      9. Pronóstico futuro con intervalos de confianza al 95%
     10. Validación cruzada rolling (expanding window)
     11. Generación de 6 gráficos de resultados
     12. Exportación de 3 CSV + 1 informe TXT
    """
    print("\n" + "═" * 70)
    print("  SARIMAX — PREDICCIÓN DEL PRECIO DEL CEMENTO")
    print("  Tesis de Grado/Posgrado — Series Temporales con Variables Exógenas")
    print("═" * 70)

    T_INICIO = time.time()

    # ── Paso 1: Hardware ─────────────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 1 ▸ Verificando hardware disponible")
    print(f"{'━'*60}")
    hw_info = verificar_hardware()

    # ── Paso 2-3: Carga de datos ─────────────────────────────────────────────
    df_cemento   = cargar_datos_historicos(RUTA_CEMENTO)
    df_nivel_rio = cargar_nivel_rio_completo(RUTA_NIVEL_RIO)

    # Unir nivel del río al dataset histórico del cemento mediante join temporal
    print(f"{'━'*60}")
    print("  PASO 3 ▸ Integrando nivel del río con datos históricos")
    print(f"{'━'*60}")
    # Eliminar Nivel_Rio del cemento si ya existía, para reemplazarlo con el nuevo archivo
    if "Nivel_Rio" in df_cemento.columns:
        print("  ℹ  Nivel_Rio encontrado en el CSV de cemento → se reemplaza con el nuevo archivo.")
        df_cemento = df_cemento.drop(columns=["Nivel_Rio"])
    df_hist = df_cemento.join(df_nivel_rio[["Nivel_Rio"]], how="left")
    n_nan_nivel = df_hist["Nivel_Rio"].isnull().sum()
    if n_nan_nivel > 0:
        print(f"  ⚠  {n_nan_nivel} valores de Nivel_Rio sin correspondencia → interpolación.")
        df_hist["Nivel_Rio"] = df_hist["Nivel_Rio"].interpolate(method="linear").bfill().ffill()
    print(f"  ✓  Dataset histórico integrado: {len(df_hist)} observaciones con Nivel_Rio")

    # Exógenas futuras: fechas del nivel del río posteriores al último dato histórico
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

    # ── Paso 4: Análisis exploratorio ────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 4 ▸ Análisis Exploratorio de Datos (EDA)")
    print(f"{'━'*60}")

    # Tests de estacionariedad
    est_precio = test_estacionariedad(endogena_Y, "Precio del Cemento")
    est_nivel  = test_estacionariedad(df_hist["Nivel_Rio"], "Nivel del Río")

    # Gráficos EDA
    grafico_01_series_temporales(df_hist, DIR_GRAFICOS)
    grafico_02_descomposicion(endogena_Y, PERIODO_ESTACIONAL, DIR_GRAFICOS)
    grafico_03_acf_pacf(endogena_Y, lags=40, dir_salida=DIR_GRAFICOS)
    grafico_04_correlacion(df_hist, DIR_GRAFICOS)
    grafico_05_patron_estacional(endogena_Y, DIR_GRAFICOS)

    # ── Paso 5: División train/test ──────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 5 ▸ División Train / Test")
    print(f"{'━'*60}")

    Y_train = endogena_Y.iloc[:-N_TEST]
    Y_test  = endogena_Y.iloc[-N_TEST:]
    X_train = exogenas_X.iloc[:-N_TEST]
    X_test  = exogenas_X.iloc[-N_TEST:]

    print(f"  Train : {len(Y_train)} obs  ({Y_train.index[0].strftime('%Y-%m')} → "
          f"{Y_train.index[-1].strftime('%Y-%m')})")
    print(f"  Test  : {len(Y_test)} obs   ({Y_test.index[0].strftime('%Y-%m')} → "
          f"{Y_test.index[-1].strftime('%Y-%m')})")

    # ── Paso 6: Búsqueda de parámetros ──────────────────────────────────────
    best_order, best_seasonal_order, _ = buscar_parametros_auto(
        Y_train, X_train, m=PERIODO_ESTACIONAL,
    )

    # ── Paso 7: Modelo de validación ─────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 7 ▸ Validación sobre conjunto de prueba")
    print(f"{'━'*60}")

    res_train = entrenar_sarimax(
        Y_train, X_train, best_order, best_seasonal_order,
        nombre=f"Modelo de Validación SARIMAX{best_order}x{best_seasonal_order}",
    )

    # Predicción sobre el período de prueba
    forecast_obj_test = res_train.get_forecast(steps=N_TEST, exog=X_test)
    test_pred = forecast_obj_test.predicted_mean
    test_pred.index = Y_test.index

    metricas_test = calcular_metricas(Y_test, test_pred, "Conjunto de Prueba (Test Set)")

    # ── Paso 8: Modelo final (todos los datos) ───────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 8 ▸ Modelo final (re-entrenado con todos los datos)")
    print(f"{'━'*60}")

    final_res = entrenar_sarimax(
        endogena_Y, exogenas_X, best_order, best_seasonal_order,
        nombre=f"Modelo Final SARIMAX{best_order}x{best_seasonal_order}",
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

    # Asignar índice temporal del período futuro
    forecast_media.index = df_future_exog.index
    forecast_ic.index    = df_future_exog.index

    forecast_df = pd.DataFrame({
        "Media"       : forecast_media.values,
        "IC_Inferior" : forecast_ic.iloc[:, 0].values,
        "IC_Superior" : forecast_ic.iloc[:, 1].values,
        "Nivel_Rio_Exog" : df_future_exog["Nivel_Rio"].values,
    }, index=df_future_exog.index)
    forecast_df.index.name = "Fecha"

    print("\n  Pronóstico futuro:")
    print(forecast_df.round(4).to_string())

    # ── Paso 10: Validación rolling ──────────────────────────────────────────
    n_initial_rolling = max(int(len(Y_train) * 0.75), 24)
    df_rolling = validacion_rolling(
        endogena_Y.iloc[:-N_TEST],
        exogenas_X.iloc[:-N_TEST],
        best_order, best_seasonal_order,
        n_initial=n_initial_rolling,
        step=1,
    )
    metricas_rolling = (
        calcular_metricas(df_rolling["Real"], df_rolling["Predicho_Rolling"],
                          "Rolling Cross-Validation")
        if len(df_rolling) > 0 else {}
    )

    # ── Paso 11: Gráficos de resultados ─────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 11 ▸ Generando gráficos de resultados")
    print(f"{'━'*60}")

    grafico_06_pronostico_completo(
        Y_train, Y_test, test_pred, forecast_df, metricas_test, DIR_GRAFICOS,
    )
    grafico_07_diagnostico_residuos(final_res, DIR_GRAFICOS)
    grafico_08_comparacion_prueba(Y_test, test_pred, metricas_test, DIR_GRAFICOS)
    grafico_09_pronostico_ic(forecast_df, endogena_Y, DIR_GRAFICOS)
    grafico_10_rolling_forecast(df_rolling, endogena_Y, DIR_GRAFICOS)
    grafico_11_ajuste_vs_real(endogena_Y, final_res, DIR_GRAFICOS)

    # ── Paso 12: Exportar resultados ─────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("  PASO 12 ▸ Exportando resultados")
    print(f"{'━'*60}")

    guardar_resultados_csv(
        forecast_df, Y_test, test_pred, metricas_test, RUTA_SALIDA,
    )
    generar_informe_texto(
        metricas_test, metricas_rolling,
        best_order, best_seasonal_order,
        final_res, df_hist, forecast_df,
        hw_info, RUTA_SALIDA,
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    T_TOTAL = time.time() - T_INICIO
    print(f"\n{'═'*70}")
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'═'*70}")
    print(f"  Modelo       : SARIMAX{best_order}x{best_seasonal_order}")
    print(f"  RMSE (prueba): {metricas_test['RMSE']:.4f}")
    print(f"  MAE  (prueba): {metricas_test['MAE']:.4f}")
    print(f"  MAPE (prueba): {metricas_test['MAPE']:.2f} %")
    print(f"  R²   (prueba): {metricas_test['R2']:.4f}")
    print(f"  Tiempo total : {T_TOTAL:.1f} s ({T_TOTAL/60:.1f} min)")
    print(f"  Gráficos en  : {DIR_GRAFICOS}")
    print(f"  Resultados en: {RUTA_SALIDA}")
    print(f"{'═'*70}\n")

    return {
        "modelo"          : final_res,
        "forecast"        : forecast_df,
        "metricas_test"   : metricas_test,
        "metricas_rolling": metricas_rolling,
        "parametros"      : {
            "order"         : best_order,
            "seasonal_order": best_seasonal_order,
        },
    }


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    resultados = main()
