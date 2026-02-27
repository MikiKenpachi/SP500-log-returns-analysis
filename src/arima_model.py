"""
arima_model.py
--------------
ARIMA model za predikciju log-returns S&P 500.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm


def grid_search_arima(series:  pd.Series,
                      max_p:   int = 4,
                      max_q:   int = 4,
                      d:       int = 0,
                      ic:      str = 'aic') -> pd.DataFrame:
    """
    Eksplicitni grid search kombinacija ARIMA(p, d, q).
    
    Umesto da slepo verujemo auto_arima, ovde sami prolazimo kroz
    kombinacije p i q (uz fiksiran d=0 iz ADF testa) i poredimo
    AIC/BIC vrednosti. Ovo daje bolji uvid u stabilnost odabira.
    
    Preporučena upotreba:
      1. Pokrenite suggest_arima_params() iz stationarity.py za heuristiku
      2. Ograničite pretragu oko tih vrednosti (npr. max_p=4, max_q=4)
      3. Uporedite top modele – ako su AIC vrednosti bliske, preferirati
         model sa manjim p+q (radi manje složenosti modela)
    
    Heuristike za interpretaciju tabele:
      - ACF nagli pad posle lag q  → čist MA(q) model
      - PACF nagli pad posle lag p → čist AR(p) model
      - Oba postepeni pad          → ARMA(p,q), probajte (1,1), (2,1) itd.
    """
    print(f"Grid search ARIMA(p,{d},q) za p∈[0,{max_p}], q∈[0,{max_q}]...")
    print(f"Kriterijum odabira: {ic.upper()}\n")

    rezultati = []

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue  # ARIMA(0,0,0) nema smisla
            try:
                model  = ARIMA(series.dropna(), order=(p, d, q)).fit()
                aic    = model.aic
                bic    = model.bic
                # Ljung-Box na rezidualima – da li su reziduali beli šum?
                lb_p = acorr_ljungbox(model.resid, lags=[10], return_df=True)['lb_pvalue'].values[0]
                konvergirao = True
            except Exception:
                aic, bic, lb_p, konvergirao = np.nan, np.nan, np.nan, False

            rezultati.append({
                "p": p, "d": d, "q": q,
                "AIC":       round(aic,  2) if not np.isnan(aic)  else np.nan,
                "BIC":       round(bic,  2) if not np.isnan(bic)  else np.nan,
                "LB_p(10)":  round(lb_p, 4) if not np.isnan(lb_p) else np.nan,
                "konvergirao": konvergirao,
            })

    df = pd.DataFrame(rezultati).dropna(subset=['AIC', 'BIC'])
    df = df[df['konvergirao']].drop(columns='konvergirao')
    df = df.sort_values(ic.upper()).reset_index(drop=True)

    print(f"Top 10 modela po {ic.upper()}:")
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    print(f"\n→ Optimalni po {ic.upper()}: ARIMA({int(best.p)},{d},{int(best.q)})")
    print(f"  AIC={best.AIC}, BIC={best.BIC}")
    print(f"  Napomena: ako je LB_p(10) < 0.05, reziduali NISU beli šum.")

    return df


def auto_arima_search(series:   pd.Series,
                      max_p:    int  = 5,
                      max_q:    int  = 5,
                      max_d:    int  = 1,
                      seasonal: bool = False,
                      ic:       str  = 'aic',
                      verbose:  bool = True) -> dict:
    """
    Traži optimalne parametre (p, d, q) koristeći auto_arima.
    Prolazi kroz kombinacije i bira model sa najmanjim AIC/BIC.
    """
    print(f"Pokrećem auto_arima (kriterijum: {ic.upper()}, max_p={max_p}, max_q={max_q})...")

    model = pm.auto_arima(
        series.dropna(),
        start_p=0, max_p=max_p,
        start_q=0, max_q=max_q,
        max_d=max_d,
        seasonal=seasonal,
        information_criterion=ic,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=verbose,
    )

    p, d, q = model.order

    print(f"\nOptimalni model: ARIMA({p},{d},{q})")
    print(f"  AIC: {round(model.aic(), 4)}")
    print(f"  BIC: {round(model.bic(), 4)}")

    return {
        "p":     p,
        "d":     d,
        "q":     q,
        "order": (p, d, q),
        "aic":   round(model.aic(), 4),
        "bic":   round(model.bic(), 4),
        "model": model,
    }


def fit_arima(series: pd.Series, order: tuple) -> ARIMA:
    """Trenira ARIMA model zadatog reda (p,d,q)."""
    p, d, q = order
    print(f"Treniram ARIMA({p},{d},{q}) na {len(series)} opservacija...")

    fitted = ARIMA(series.dropna(), order=order).fit()

    print(f"  AIC: {round(fitted.aic, 4)}")
    print(f"  BIC: {round(fitted.bic, 4)}")

    return fitted


def walk_forward_predict(train: pd.Series,
                         test:  pd.Series,
                         order: tuple,
                         refit: bool = False) -> pd.DataFrame:
    """
    Walk-forward validacija – simulira realnu upotrebu modela.
    
    Za svaki dan u test skupu:
      1. Treniramo model na svim dostupnim podacima do tog dana
      2. Predviđamo jedan korak unapred
      3. Dodajemo pravu vrednost u istoriju i ponavljamo
    
    Ovako garantujemo da model nikad ne vidi podatke iz budućnosti.
    """
    p, d, q = order
    print(f"Walk-forward validacija – ARIMA({p},{d},{q}), {len(test)} predikcija")

    istorija = list(train.dropna().values)
    predikcije, prave, lower_ci, upper_ci, datumi = [], [], [], [], []

    for i, (datum, prava_vrednost) in enumerate(test.items()):

        # fitujemo model na trenutnoj istoriji
        if refit or i == 0:
            model  = ARIMA(istorija, order=order)
            fitted = model.fit()

        # predikcija za sledeći korak
        forecast = fitted.get_forecast(steps=1)
        pred = float(np.asarray(forecast.predicted_mean).flat[0])
        ci   = np.asarray(forecast.conf_int(alpha=0.05))

        predikcije.append(pred)
        prave.append(prava_vrednost)
        lower_ci.append(float(ci.flat[0]))
        upper_ci.append(float(ci.flat[1]))
        datumi.append(datum)

        # dodajemo pravu vrednost i refitujemo za sledeći korak
        istorija.append(prava_vrednost)
        if not refit and i < len(test) - 1:
            model  = ARIMA(istorija, order=order)
            fitted = model.fit()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)} predikcija završeno...")

    print("  Walk-forward validacija završena.")

    return pd.DataFrame({
        'y_true':   prave,
        'y_pred':   predikcije,
        'residual': [p - t for p, t in zip(prave, predikcije)],
        'lower_95': lower_ci,
        'upper_95': upper_ci,
    }, index=pd.DatetimeIndex(datumi))


def dijagnostika_reziduala(rezultati: pd.DataFrame,
                           model_name: str = "ARIMA") -> dict:
    """
    Ljung-Box test na rezidualima i kvadriranim rezidualima.
    
    - Reziduali: testiramo da li su beli šum (H0: nema autokorelacije)
    - Kvadrirani reziduali: testiramo ARCH efekat (volatility clustering)
      Ako postoji → opravdava primenu GARCH modela
    """
    reziduali = rezultati['residual'].dropna()

    lb_rez  = acorr_ljungbox(reziduali,      lags=[10, 20], return_df=True)
    lb_arch = acorr_ljungbox(reziduali ** 2, lags=[10, 20], return_df=True)

    print(f"{'─' * 55}")
    print(f"  Dijagnostika reziduala – {model_name}")
    print(f"{'─' * 55}")
    print(f"\n  Ljung-Box (reziduali) – H0: nema autokorelacije")
    print(lb_rez.to_string())
    print(f"\n  Ljung-Box (reziduali²) – H0: nema ARCH efekta")
    print(lb_arch.to_string())

    arch_efekat = any(p < 0.05 for p in lb_arch['lb_pvalue'].values)
    print(f"\n  ARCH efekat: {'DA → preporučuje se GARCH' if arch_efekat else 'NE'}")
    print(f"{'─' * 55}\n")

    return {
        "ljungbox_reziduali": lb_rez,
        "ljungbox_arch":      lb_arch,
        "arch_efekat":        arch_efekat,
    }


def plot_arima_predictions(rezultati:  pd.DataFrame,
                           model_name: str = "ARIMA",
                           n_prikaz:   int = None) -> None:
    """Vizualizuje predikcije vs stvarne vrednosti sa intervalom poverenja."""
    df = rezultati.iloc[-n_prikaz:] if n_prikaz else rezultati

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df.index, df['y_true'],
            color='#2c3e50', linewidth=1.0, label='Stvarne vrednosti')
    ax.plot(df.index, df['y_pred'],
            color='#e74c3c', linewidth=1.0, linestyle='--',
            label=f'Predikcija ({model_name})')
    ax.fill_between(df.index, df['lower_95'], df['upper_95'],
                    color='#e74c3c', alpha=0.15, label='Interval poverenja (95%)')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{model_name} – walk-forward predikcija log-returns',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Log-return')
    ax.set_xlabel('Datum')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_arima_residuals(rezultati: pd.DataFrame, model_name: str = "ARIMA") -> None:
    """
    Reziduali i kvadrirani reziduali kroz vreme.
    Kvadrirani reziduali pokazuju da li postoji volatility clustering (ARCH efekat).
    """
    rez = rezultati['residual']

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(rez.index, rez.values, color='#2980b9', linewidth=0.8)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[0].set_title(f'{model_name} – reziduali kroz vreme',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Rezidual')

    axes[1].plot(rez.index, rez.values ** 2, color='#e67e22', linewidth=0.8)
    axes[1].set_title('Kvadrirani reziduali – provera ARCH efekta',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Rezidual²')
    axes[1].set_xlabel('Datum')

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_reziduali_za_garch(rezultati: pd.DataFrame) -> pd.Series:
    """
    Vraća rezidualne greške ARIMA modela.
    Ova serija se prosleđuje u garch_model.py kao ulaz.
    """
    return rezultati['residual'].dropna()
