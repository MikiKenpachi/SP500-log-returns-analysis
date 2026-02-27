"""
garch_model.py
--------------
GARCH model za predikciju kondicionalne volatilnosti log-returns S&P 500.

GARCH(p, q) – Generalized AutoRegressive Conditional Heteroskedasticity
(Bollerslev, 1986) modeluje heteroskedastičnost (promenljivu varijansu)
koja je karakteristična za finansijske serije (volatility clustering).

Ulaz: reziduali ARIMA modela (ili direktno log-returns)
Izlaz: predikcija kondicionalne standardne devijacije (volatilnosti)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model
from arch.univariate import GARCH
from statsmodels.stats.diagnostic import acorr_ljungbox


# ── Pomoćne funkcije ─────────────────────────────────────────────────────────

def skaliraj_za_arch(series: pd.Series, faktor: float = 100.0) -> pd.Series:
    """
    ARCH biblioteka numerički je stabilnija kad su vrednosti u rangu celog broja.
    Log-returns su reda veličine 0.01, pa množimo sa 100 (-> %).
    Sve metrike i grafici moraju uzeti ovo u obzir.
    """
    return series * faktor


# ── Fitovanje modela ─────────────────────────────────────────────────────────

def fit_garch(series: pd.Series,
              p: int = 1,
              q: int = 1,
              dist: str = 'normal',
              mean: str = 'Zero') -> object:
    """
    Trenira GARCH(p,q) model na zadatoj seriji.

    Parametri:
      series : log-returns ili ARIMA reziduali (pd.Series)
      p      : red ARCH člana (efekti prošlih šokova na varijansu)
      q      : red GARCH člana (efekti prošlih varijanci)
      dist   : raspodela inovacija – 'normal', 'studentst', 'skewt'
      mean   : model za srednju vrednost – 'Zero', 'Constant', 'AR'

    Napomena: za finansijske log-returns preporučuje se dist='studentst'
    jer leptokurtična raspodela bolje opisuje debele repove.
    """
    skalirana = skaliraj_za_arch(series)
    print(f"Treniram GARCH({p},{q}) | dist={dist} | mean={mean}")
    print(f"  N opservacija : {len(skalirana)}")

    model = arch_model(
        skalirana,
        vol='Garch',
        p=p,
        q=q,
        dist=dist,
        mean=mean,
    )
    fitted = model.fit(disp='off', show_warning=False)

    print(f"  Log-likelihood : {fitted.loglikelihood:.4f}")
    print(f"  AIC            : {fitted.aic:.4f}")
    print(f"  BIC            : {fitted.bic:.4f}")

    return fitted


def uporedi_garch_distribucije(series: pd.Series,
                                p: int = 1,
                                q: int = 1) -> pd.DataFrame:
    """
    Poredi GARCH(p,q) sa različitim raspodelamama inovacija po AIC/BIC.
    Preporučen izbor: raspodela sa najnižim AIC.
    """
    distribucije = ['normal', 'studentst', 'skewt']
    rezultati = []

    for dist in distribucije:
        try:
            res = fit_garch(series, p=p, q=q, dist=dist)
            rezultati.append({
                'Distribucija': dist,
                'LogL':  round(res.loglikelihood, 4),
                'AIC':   round(res.aic, 4),
                'BIC':   round(res.bic, 4),
            })
        except Exception as e:
            print(f"  {dist}: greška – {e}")

    df = pd.DataFrame(rezultati).sort_values('AIC').reset_index(drop=True)
    print("\nPoređenje distribucija GARCH inovacija:")
    print(df.to_string(index=False))
    best = df.iloc[0]['Distribucija']
    print(f"\n→ Preporučena distribucija: {best}")
    return df


# ── Walk-forward validacija ──────────────────────────────────────────────────

def walk_forward_garch(train: pd.Series,
                       test: pd.Series,
                       p: int = 1,
                       q: int = 1,
                       dist: str = 'studentst',
                       horizon: int = 1) -> pd.DataFrame:
    """
    Walk-forward validacija GARCH modela – procena kondicionalne volatilnosti.

    Za razliku od ARIMA (koji predviđa srednju vrednost),
    GARCH predviđa VARIJANSU (kvadrat volatilnosti) za sledeći dan.

    Strategija: fitujemo model JEDNOM na trening skupu i dobijamo
    optimalne parametre (ω, α, β). Za svaki korak u test skupu
    koristimo `model.fix(params)` – ovo ažurira kondicionu varijansu
    sa novom opservacijom BEZ ponovne optimizacije parametara.

    Prednosti fix() pristupa vs refitovanje:
      - Brzina: O(N) umesto O(N²) – ~240× brže za test skup od 240 dana
      - Realističnost: parametri se ne menjaju između koraka (konzistentno
        sa pretpostavkom da su parametri stabilni u kratkom periodu)
      - Numerička stabilnost: izbegava se potencijalna nekonvergencija
        pri svakom refitovanju na proširenom skupu

    Ulaz:
      train : ARIMA reziduali trening skupa (ne originalni log-returns!)
      test  : ARIMA reziduali test skupa
    """
    print(f"Walk-forward GARCH({p},{q}) | {len(test)} dana")
    skaliran_train = skaliraj_za_arch(train)
    skaliran_test  = skaliraj_za_arch(test)

    # ── Korak 1: Fitujemo JEDNOM na trening skupu ────────────────────────────
    print("  Fitovanje na trening skupu (jedanput)...")
    init_model = arch_model(skaliran_train, vol='Garch', p=p, q=q,
                            dist=dist, mean='Zero')
    init_fitted = init_model.fit(disp='off', show_warning=False)
    params = init_fitted.params
    print(f"  Parametri fiksirani: {dict(round(params, 6))}")

    # ── Korak 2: Walk-forward sa fix(params) ─────────────────────────────────
    historija = list(skaliran_train.values)
    vol_pred, vol_true, datumi = [], [], []

    for i, (datum, prava) in enumerate(skaliran_test.items()):

        # Primenjujemo fiksirane parametre na proširenu istoriju
        # fix() ažurira kondicionu varijansu bez reoptimizacije – O(N) složenost
        model_step = arch_model(historija, vol='Garch', p=p, q=q,
                                dist=dist, mean='Zero')
        res_step = model_step.fix(params)

        # Prognoza volatilnosti za sledeći korak
        forecast = res_step.forecast(horizon=horizon, reindex=False)
        var_pred = float(forecast.variance.values[-1, 0])
        sigma_pred = np.sqrt(max(var_pred, 0))  # std dev u % skali

        # "Prava" volatilnost = |rezidual_t| (proxy – standardna praksa)
        sigma_true = abs(float(prava))

        vol_pred.append(sigma_pred)
        vol_true.append(sigma_true)
        datumi.append(datum)

        historija.append(float(prava))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)} predikcija završeno...")

    print("  Walk-forward GARCH validacija završena.")

    df = pd.DataFrame({
        'sigma_true':  vol_true,    # % skala (×100)
        'sigma_pred':  vol_pred,    # % skala (×100)
        'sigma_true_pct': [v / 100 for v in vol_true],   # originalna skala
        'sigma_pred_pct': [v / 100 for v in vol_pred],
    }, index=pd.DatetimeIndex(datumi))

    return df


def garch_jednokratna_prognoza(series: pd.Series,
                                p: int = 1,
                                q: int = 1,
                                dist: str = 'studentst',
                                horizon: int = 10) -> pd.DataFrame:
    """
    Jednokratna prognoza volatilnosti za narednih `horizon` dana.
    Trenira na celoj prosleđenoj seriji.

    Korisno za vizualizaciju: kako se prognoza volatilnosti
    razvija u narednom periodu.
    """
    skalirana = skaliraj_za_arch(series)
    model  = arch_model(skalirana, vol='Garch', p=p, q=q, dist=dist, mean='Zero')
    fitted = model.fit(disp='off', show_warning=False)
    forecast = fitted.forecast(horizon=horizon, reindex=False)

    var_vals = forecast.variance.values[-1]
    sigma_vals = np.sqrt(np.maximum(var_vals, 0)) / 100  # nazad na originale

    df = pd.DataFrame({
        'horizon':     range(1, horizon + 1),
        'sigma_pred':  sigma_vals,
    })
    return df, fitted


# ── Dijagnostika ─────────────────────────────────────────────────────────────

def dijagnostika_garch(fitted_model: object,
                       model_name: str = "GARCH(1,1)") -> dict:
    """
    Dijagnostika standardizovanih reziduala GARCH modela.

    Standardizovani reziduali: ε_t / σ_t
    Ako je model dobar → trebalo bi biti beli šum i bez ARCH efekta.

    Ljung-Box na standardizovanim rezidualima → linearnost
    Ljung-Box na kvadratima std. reziduala   → heteroskedastičnost
    """
    std_resid = fitted_model.std_resid.dropna()

    lb_std   = acorr_ljungbox(std_resid,      lags=[5, 10, 20], return_df=True)
    lb_sq    = acorr_ljungbox(std_resid ** 2, lags=[5, 10, 20], return_df=True)

    print(f"{'─' * 60}")
    print(f"  Dijagnostika – {model_name}")
    print(f"{'─' * 60}")
    print(f"\n  Ljung-Box (standardizovani reziduali) – H0: beli šum")
    print(lb_std.to_string())
    print(f"\n  Ljung-Box (kvadrati std. reziduala) – H0: nema ARCH efekta")
    print(lb_sq.to_string())

    arch_ostao = any(p < 0.05 for p in lb_sq['lb_pvalue'].values)
    print(f"\n  Preostali ARCH efekat: {'DA – model nije uhvatio svu heteroskedastičnost' if arch_ostao else 'NE – GARCH je dobro uhvatio volatilnost'}")
    print(f"{'─' * 60}\n")

    return {
        'lb_std_resid': lb_std,
        'lb_sq_resid':  lb_sq,
        'arch_ostao':   arch_ostao,
    }


# ── Vizualizacije ─────────────────────────────────────────────────────────────

def plot_kondiciona_volatilnost(fitted_model: object,
                                series: pd.Series,
                                model_name: str = "GARCH(1,1)") -> None:
    """
    Prikazuje log-returns i uslovne standardne devijacije (σ_t)
    iz fitovanog GARCH modela kroz vreme.

    Uslovna. std. dev. = prognoza volatilnosti za svaki dan na osnovu
    prethodnih informacija → prikazuje dinamiku rizika kroz vreme.
    """
    kondiciona_var = fitted_model.conditional_volatility  # u % skali
    sigma_t = kondiciona_var / 100  # nazad na originale

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Gornji: log-returns
    axes[0].plot(series.index, series.values,
                 color='#2c3e50', linewidth=0.7, alpha=0.8, label='Log-return')
    axes[0].fill_between(sigma_t.index, -2 * sigma_t.values, 2 * sigma_t.values,
                          color='#e74c3c', alpha=0.2, label='±2σ interval uslovne volatilnosti')
    axes[0].set_title(f'Log-returns S&P 500 sa intervalom uslovne volatilnosti – {model_name}',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Log-return')
    axes[0].legend(loc='upper left')

    # Donji: Uslovna volatilnost
    axes[1].plot(sigma_t.index, sigma_t.values,
                 color='#e74c3c', linewidth=1.0, label='Uslovna σ_t')
    axes[1].set_title('Uslovna volatilnost (σ_t) – procena rizika kroz vreme',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('σ_t (log-return skala)')
    axes[1].set_xlabel('Datum')
    axes[1].legend()

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_garch_forecast(df_wf: pd.DataFrame,
                        series_true: pd.Series = None,
                        model_name: str = "GARCH(1,1)") -> None:
    """
    Walk-forward prognoza volatilnosti vs proxy stvarne volatilnosti.

    Napomena: GARCH predviđa buduću σ_t, a "prava" volatilnost za
    jedan dan nije direktno merljiva. Koristimo |r_t| (apsolutni prinos)
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df_wf.index, df_wf['sigma_true_pct'],
            color='#2c3e50', linewidth=0.8, alpha=0.7,
            label='|r_t| – apsolutni prinos')
    ax.plot(df_wf.index, df_wf['sigma_pred_pct'],
            color='#e74c3c', linewidth=1.2, linestyle='--',
            label=f'Prognozirana σ_t ({model_name})')

    ax.set_title(f'{model_name} – Walk-forward prognoza volatilnosti (test skup)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Volatilnost (log-return skala)')
    ax.set_xlabel('Datum')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_garch_horizon_forecast(df_horizon: pd.DataFrame,
                                 model_name: str = "GARCH(1,1)") -> None:
    """
    Prognoza volatilnosti: kako se σ_t razvija kroz
    narednih horizon dana. GARCH prognoza konvergira ka bezuslovnoj
    (dugoročnoj) volatilnosti – tzv. mean reversion volatilnosti.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df_horizon['horizon'], df_horizon['sigma_pred'],
            marker='o', color='#e74c3c', linewidth=1.5)
    ax.fill_between(df_horizon['horizon'],
                    df_horizon['sigma_pred'] * 0.85,
                    df_horizon['sigma_pred'] * 1.15,
                    alpha=0.2, color='#e74c3c', label='±15% interval')

    ax.set_title(f'{model_name} – Višestepena prognoza volatilnosti',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Koraci unapred (dani)')
    ax.set_ylabel('Prognozirana σ (log-return skala)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_standardizovani_reziduali(fitted_model: object,
                                    model_name: str = "GARCH(1,1)") -> None:
    """
    Dijagnostički grafik standardizovanih reziduala (ε_t / σ_t).
    Ako je model dobar → trebalo bi ličiti na beli šum bez klastera.
    """
    std_resid = fitted_model.std_resid.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(std_resid.index, std_resid.values,
                 color='#2980b9', linewidth=0.7)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[0].axhline(2, color='orange', linestyle=':', linewidth=0.8)
    axes[0].axhline(-2, color='orange', linestyle=':', linewidth=0.8)
    axes[0].set_title(f'{model_name} – Standardizovani reziduali (ε_t / σ_t)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('std. rezidual')

    axes[1].plot(std_resid.index, std_resid.values ** 2,
                 color='#e67e22', linewidth=0.7)
    axes[1].set_title('Kvadrirani std. reziduali – preostali ARCH efekat',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('(std. rezidual)²')
    axes[1].set_xlabel('Datum')

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ── Metrike za volatilnost ────────────────────────────────────────────────────

def garch_metrike(df_wf: pd.DataFrame, model_name: str = "GARCH(1,1)") -> pd.DataFrame:
    """
    Metrike prognoze volatilnosti:
      - MAE volatilnosti  : prosečna greška u prognozi σ
      - RMSE volatilnosti : kažnjava veće greške
      - QLIKE             : kvazi-likelihood metrika (standardna u literaturi)

    Napomena: Direktno poređenje GARCH-a i ARIMA-e po istim metrikama
    nije smisleno jer predviđaju različite veličine (σ vs μ).
    """
    true_sq  = df_wf['sigma_true_pct'] ** 2
    pred_var = df_wf['sigma_pred_pct'] ** 2

    mae_vol  = float(np.mean(np.abs(df_wf['sigma_true_pct'] - df_wf['sigma_pred_pct'])))
    rmse_vol = float(np.sqrt(np.mean((df_wf['sigma_true_pct'] - df_wf['sigma_pred_pct']) ** 2)))

    # QLIKE: standardna loss funkcija za poređenje volatilnostnih modela
    # QLIKE = E[σ²_pred/σ²_true - log(σ²_pred/σ²_true) - 1]
    ratio = pred_var / (true_sq + 1e-10)
    qlike = float(np.mean(ratio - np.log(ratio + 1e-10) - 1))

    df_m = pd.DataFrame([{
        'Model':       model_name,
        'MAE_vol':     round(mae_vol,  6),
        'RMSE_vol':    round(rmse_vol, 6),
        'QLIKE':       round(qlike,    6),
    }])
    print(f"\nMetrike prognoze volatilnosti – {model_name}:")
    print(df_m.to_string(index=False))

    return df_m
