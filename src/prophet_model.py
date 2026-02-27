"""
prophet_model.py
----------------
Facebook Prophet model za predikciju log-returns S&P 500.

Prophet je razvijen u Meta-i (Taylor & Letham, 2018) i dizajniran
za poslovne vremenske serije sa jakim sezonalnostima i praznicima.
Za log-returns finansijskih indeksa (koji su blizu belog šuma) Prophet
neće biti superioran, ali pruža alternativan pristup za poređenje.

Napomena: Prophet interno koristi Stanov model trenda i Furijeov niz
za sezonalnost – ne oslanja se na stacionarnost kao ARIMA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
except ImportError:
    raise ImportError("Prophet nije instaliran. Pokrenite: pip install prophet")

from statsmodels.stats.diagnostic import acorr_ljungbox


# ── Konverzija u Prophet format ───────────────────────────────────────────────

def pripremi_prophet_df(series: pd.Series) -> pd.DataFrame:
    """
    Prophet zahteva DataFrame sa kolonama 'ds' (datum) i 'y' (vrednost).
    Konvertujemo pd.Series sa DatetimeIndex u ovaj format.
    """
    df = pd.DataFrame({
        'ds': series.index,
        'y':  series.values,
    }).reset_index(drop=True)
    df['ds'] = pd.to_datetime(df['ds'])
    return df


# ── Fitovanje modela ─────────────────────────────────────────────────────────

def fit_prophet(train: pd.Series,
                yearly_seasonality: bool = False,
                weekly_seasonality: bool = True,
                daily_seasonality:  bool = False,
                changepoint_prior_scale: float = 0.05,
                seasonality_prior_scale: float = 10.0,
                interval_width: float = 0.95) -> Prophet:
    """
    Trenira Prophet model na trening seriji.

    Parametri:
      yearly_seasonality      : godišnja sezonalnost (Furijeov red)
      weekly_seasonality      : nedeljni efekti (petak/ponedeljak)
      daily_seasonality       : intra-dnevna (nije relevantno za dnevne podatke)
      changepoint_prior_scale : fleksibilnost trenda (veće = fleksibilniji trend)
      seasonality_prior_scale : jačina sezonalnih komponenti
      interval_width          : širina intervala poverenja

    Za log-returns:
      - yearly_seasonality=False jer log-returns nemaju godišnju sezonalnost
      - weekly_seasonality=True jer postoje efekti dana u nedelji (Monday effect)
      - changepoint_prior_scale=0.05 jer ne očekujemo jake promene trenda
    """
    df_train = pripremi_prophet_df(train)

    #print(f"Treniram Prophet model na {len(df_train)} opservacija...")
    #print(f"  weekly_seasonality      = {weekly_seasonality}")
    #print(f"  yearly_seasonality      = {yearly_seasonality}")
    #print(f"  changepoint_prior_scale = {changepoint_prior_scale}")

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        interval_width=interval_width,
    )

    model.fit(df_train)
    print("  Treniranje završeno.")
    return model


# ── Walk-forward validacija ──────────────────────────────────────────────────

def walk_forward_prophet(train: pd.Series,
                          test: pd.Series,
                          refit_interval: int = 20,
                          **prophet_kwargs) -> pd.DataFrame:
    """
    Walk-forward validacija Prophet modela.

    Prophet je sporiji od ARIMA pa ne refitujemo svaki dan – refitujemo
    svakih `refit_interval` dana. Ovo je razumna aproksimacija jer
    Prophet ne menja drastično predikcije za jedan dan.

    Parametri:
      train           : trening serija log-returns
      test            : test serija log-returns
      refit_interval  : koliko dana između refitovanja (default: 20)
      **prophet_kwargs: prosleđuju se u fit_prophet()
    """
    print(f"\nWalk-forward Prophet | {len(test)} predikcija | refit svakih {refit_interval} dana")

    istorija = train.copy()
    predikcije, prave, lower_ci, upper_ci, datumi = [], [], [], [], []
    model = None

    for i, (datum, prava_vrednost) in enumerate(test.items()):

        # Refitujemo svakih refit_interval dana
        if i % refit_interval == 0:
            model = fit_prophet(istorija, **prophet_kwargs)

        # Predikcija za sledeći dan
        future = pd.DataFrame({'ds': [datum]})
        forecast = model.predict(future)

        pred      = float(forecast['yhat'].values[0])
        lower     = float(forecast['yhat_lower'].values[0])
        upper     = float(forecast['yhat_upper'].values[0])

        predikcije.append(pred)
        prave.append(prava_vrednost)
        lower_ci.append(lower)
        upper_ci.append(upper)
        datumi.append(datum)

        # Dodajemo pravu vrednost u istoriju
        nova = pd.Series([prava_vrednost], index=[datum])
        istorija = pd.concat([istorija, nova])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)} predikcija završeno...")

    print("  Walk-forward Prophet validacija završena.")

    return pd.DataFrame({
        'y_true':   prave,
        'y_pred':   predikcije,
        'residual': [p - t for p, t in zip(prave, predikcije)],
        'lower_95': lower_ci,
        'upper_95': upper_ci,
    }, index=pd.DatetimeIndex(datumi))


# ── Komponente modela ─────────────────────────────────────────────────────────

def plot_prophet_komponente(model: Prophet,
                             train: pd.Series,
                             model_name: str = "Prophet") -> None:
    """
    Prikazuje komponente Prophet modela: trend i sezonalnost.
    Ovo je jedna od ključnih prednosti Propheta – interpretabilnost komponenti.
    """
    df_train = pripremi_prophet_df(train)
    forecast = model.predict(df_train)

    fig = model.plot_components(forecast)
    fig.suptitle(f'{model_name} – komponente modela (trend + sezonalnost)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_prophet_prognoza(model: Prophet,
                           train: pd.Series,
                           test: pd.Series,
                           model_name: str = "Prophet") -> None:
    """
    Prophet interna vizualizacija prognoze za ceo period.
    Prikazuje fit na trening skupu i predikcije na test skupu.
    """
    df_all = pripremi_prophet_df(pd.concat([train, test]))
    forecast = model.predict(df_all)

    fig = model.plot(forecast)
    ax = fig.axes[0]
    ax.axvline(test.index[0], color='red', linestyle='--',
               linewidth=1.5, label='Početak test skupa')
    ax.set_title(f'{model_name} – prognoza log-returns', fontsize=13, fontweight='bold')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Log-return')
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Vizualizacije (uniformisane sa arima_model.py stilom) ─────────────────────

def plot_prophet_predictions(rezultati: pd.DataFrame,
                              model_name: str = "Prophet",
                              n_prikaz: int = None) -> None:
    """Vizualizuje Prophet predikcije vs stvarne vrednosti."""
    df = rezultati.iloc[-n_prikaz:] if n_prikaz else rezultati

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df.index, df['y_true'],
            color='#2c3e50', linewidth=1.0, label='Stvarne vrednosti')
    ax.plot(df.index, df['y_pred'],
            color='#27ae60', linewidth=1.0, linestyle='--',
            label=f'Predikcija ({model_name})')
    ax.fill_between(df.index, df['lower_95'], df['upper_95'],
                    color='#27ae60', alpha=0.15, label='Interval poverenja (95%)')

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


def plot_prophet_residuals(rezultati: pd.DataFrame,
                            model_name: str = "Prophet") -> None:
    """Reziduali i kvadrirani reziduali Prophet modela."""
    rez = rezultati['residual']

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(rez.index, rez.values, color='#27ae60', linewidth=0.8)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[0].set_title(f'{model_name} – reziduali kroz vreme',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Rezidual')

    axes[1].plot(rez.index, rez.values ** 2, color='#f39c12', linewidth=0.8)
    axes[1].set_title('Kvadrirani reziduali – provera ARCH efekta',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Rezidual²')
    axes[1].set_xlabel('Datum')

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ── Dijagnostika reziduala ────────────────────────────────────────────────────

def dijagnostika_prophet(rezultati: pd.DataFrame,
                          model_name: str = "Prophet") -> dict:
    """Ljung-Box dijagnostika reziduala Prophet modela."""
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
    
    print(f"{'─' * 55}\n")

    return {
        'ljungbox_reziduali': lb_rez,
        'ljungbox_arch':      lb_arch,
        'arch_efekat':        arch_efekat,
    }
