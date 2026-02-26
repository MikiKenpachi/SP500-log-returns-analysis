"""
evaluation.py
-------------
Metrike za evaluaciju predikcija vremenskih serija.

Koristimo MAE, RMSE i MASE.
MAPE je izbačena jer log-returns osciluju oko nule
što dovodi do numerički nestabilnih rezultata pri deljenju!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error – kažnjava veće greške jače od MAE."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mase(y_true: np.ndarray,
         y_pred: np.ndarray,
         y_train: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error.
    
    Poredi naš model sa naivnim prediktorom (predikcija = prethodna vrednost).
    MASE < 1 znači da je naš model bolji od naivnog.
    """
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    if naive_mae == 0:
        raise ValueError("Naivni MAE je nula – skaliranje nije moguće.")
    return mae(y_true, y_pred) / naive_mae


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_train: Optional[np.ndarray] = None,
                        model_name: str = "Model") -> pd.DataFrame:
    """
    Računa MAE, RMSE i MASE i vraća ih kao DataFrame red.
    Ako y_train nije prosleđen, MASE se preskače.
    """
    metrics = {
        "Model": model_name,
        "MAE":   round(mae(y_true, y_pred), 6),
        "RMSE":  round(rmse(y_true, y_pred), 6),
    }
    if y_train is not None:
        metrics["MASE"] = round(mase(y_true, y_pred, y_train), 4)

    return pd.DataFrame([metrics])


def compare_models(results: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Spaja listu DataFrame-ova iz compute_all_metrics u jednu tabelu.
    
    Primer:
        tabela = compare_models([arima_metrics, garch_metrics])
    """
    return pd.concat(results, ignore_index=True).set_index("Model")


def plot_predictions(y_true:  pd.Series,
                     y_pred:  pd.Series,
                     model_name: str,
                     conf_lower: Optional[pd.Series] = None,
                     conf_upper: Optional[pd.Series] = None,
                     title_suffix: str = "") -> None:
    """Vizualizuje tačne vrednosti nasuprot predikcijama, opciono sa intervalom poverenja."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(y_true.index, y_true.values,
            label="Stvarne vrednosti", color="#2c3e50", linewidth=1.2)
    ax.plot(y_pred.index, y_pred.values,
            label=f"Predikcija ({model_name})",
            color="#e74c3c", linewidth=1.2, linestyle="--")

    if conf_lower is not None and conf_upper is not None:
        ax.fill_between(y_pred.index, conf_lower.values, conf_upper.values,
                        alpha=0.2, color="#e74c3c", label="Interval poverenja (95%)")

    ax.set_title(f"{model_name} – predikcija log-returns {title_suffix}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Log-return")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_residuals(residuals: pd.Series, model_name: str) -> None:
    """Reziduali i kvadrirani reziduali kroz vreme."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    axes[0].plot(residuals.index, residuals.values,
                 color="#2980b9", linewidth=0.8)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"{model_name} – reziduali kroz vreme",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Rezidual")

    axes[1].plot(residuals.index, residuals.values ** 2,
                 color="#e67e22", linewidth=0.8)
    axes[1].set_title("Kvadrirani reziduali – provera ARCH efekta",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Rezidual²")
    axes[1].set_xlabel("Datum")

    plt.tight_layout()
    plt.show()
