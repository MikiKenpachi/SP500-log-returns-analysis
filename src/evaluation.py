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


def log_return_to_price(log_returns_pred: pd.Series,
                        poslednja_cena:   float) -> pd.Series:
    """
    Konvertuje predviđene log-returns u apsolutne cene indeksa.
    
    Matematika:
      log_return_t = ln(P_t / P_{t-1})
      ⟹ P_t = P_{t-1} * exp(log_return_t)
    
    Kumulativno za niz predikcija:
      P_t = P_0 * exp(Σ log_return_i)  za i=1..t
    
    Parametri:
      log_returns_pred : predviđeni log-returns (pd.Series sa DatetimeIndex)
      poslednja_cena   : poslednja poznata cena pre perioda predikcije (P_0)
    
    Napomena o interpretaciji:
      - Pozitivan log-return od 0.01 ≈ rast od ~1% (exp(0.01) ≈ 1.01005)
      - Negativan log-return od -0.02 ≈ pad od ~2%
      - Za male vrednosti: log_return ≈ prosta stopa prinosa
    """
    # Kumulativni log-returns od polazne tačke
    kumulativni = log_returns_pred.cumsum()
    cene = poslednja_cena * np.exp(kumulativni)
    return cene


def interpret_log_returns(log_returns_true: pd.Series,
                          log_returns_pred:  pd.Series,
                          poslednja_cena:    float,
                          model_name:        str = "ARIMA") -> pd.DataFrame:
    """
    Kompletna interpretabilna tabela predikcija.
    
    Za svaki dan prikazuje:
      - Predviđeni log-return (sirova predikcija modela)
      - Predviđenu prostu stopu prinosa (% promena) – lakša za intuiciju
      - Predviđenu cenu indeksa
      - Stvarnu cenu indeksa
      - Grešku u ceni (EUR/USD razlika)
    """
    df = pd.DataFrame({
        'log_return_stvarni':  log_returns_true.values,
        'log_return_predviđen': log_returns_pred.values,
    }, index=log_returns_true.index)

    # Prosta stopa prinosa ≈ exp(r) - 1 (lakša za intuiciju od log-returna)
    df['prosta_stopa_pred_%'] = (np.exp(df['log_return_predviđen']) - 1) * 100
    df['prosta_stopa_stvarna_%'] = (np.exp(df['log_return_stvarni']) - 1) * 100

    # Konverzija na cene
    df['cena_predviđena'] = poslednja_cena * np.exp(df['log_return_predviđen'].cumsum())
    df['cena_stvarna']    = poslednja_cena * np.exp(df['log_return_stvarni'].cumsum())
    df['greška_cene']     = df['cena_predviđena'] - df['cena_stvarna']

    return df.round(4)


def plot_price_interpretation(df_interpret: pd.DataFrame,
                               model_name:   str = "ARIMA") -> None:
    """
    Dvostruki grafik: gore cene, dole log-returns.
    Pomaže u interpretaciji predikcija – pokazuje i 'sirove' log-returns
    i njihov efekat na cenu indeksa.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Gornji panel: cena indeksa
    axes[0].plot(df_interpret.index, df_interpret['cena_stvarna'],
                 color='#2c3e50', linewidth=1.2, label='Stvarna cena')
    axes[0].plot(df_interpret.index, df_interpret['cena_predviđena'],
                 color='#e74c3c', linewidth=1.2, linestyle='--',
                 label=f'Predviđena cena ({model_name})')
    axes[0].set_title('Cena indeksa – stvarna vs predviđena', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cena indeksa')
    axes[0].legend()

    # Donji panel: log-returns
    axes[1].plot(df_interpret.index, df_interpret['log_return_stvarni'],
                 color='#2c3e50', linewidth=0.9, label='Stvarni log-return')
    axes[1].plot(df_interpret.index, df_interpret['log_return_predviđen'],
                 color='#e74c3c', linewidth=0.9, linestyle='--',
                 label=f'Predviđeni log-return ({model_name})')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_title('Log-returns – stvarni vs predviđeni', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Log-return  [≈ % promena / 100]')
    axes[1].set_xlabel('Datum')
    axes[1].legend()

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Kratak opis za čitaoce
    print("Napomena o interpretaciji log-returns:")
    print("  log_return_t = ln(P_t / P_{t-1})")
    print("  Za male vrednosti: log_return ≈ procentualna promena / 100")
    print("  Primer: log_return = 0.012 → cena porasla ~1.2%")
    print("  Primer: log_return = -0.020 → cena pala ~2.0%")
    print()
    print(f"  Opseg predviđenih log-returns: [{df_interpret['log_return_predviđen'].min():.4f}, "
          f"{df_interpret['log_return_predviđen'].max():.4f}]")
    greska_pct = (df_interpret['greška_cene'].abs() / df_interpret['cena_stvarna'] * 100).mean()
    print(f"  Prosečna apsolutna greška u ceni: {greska_pct:.2f}%")


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
