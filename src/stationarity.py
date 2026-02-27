"""
stationarity.py
---------------
Testiranje stacionarnosti i analiza ACF/PACF strukture.
Rezultati određuju parametre d, p, q za ARIMA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm  # za runs test z-statistiku


def adf_test(series: pd.Series, naziv: str = "Serija") -> dict:
    """
    Augmented Dickey-Fuller test stacionarnosti.
    H0: serija ima jedinični koren (nije stacionarna).
    Odbacujemo H0 ako je p-vrednost <= 0.05.
    """
    result = adfuller(series.dropna(), autolag='AIC')

    adf_stat  = result[0]
    p_value   = result[1]
    crit_vals = result[4]

    if p_value <= 0.01:
        zakljucak = "STACIONARNA (99%)"
        stacionarna = True
    elif p_value <= 0.05:
        zakljucak = "STACIONARNA (95%)"
        stacionarna = True
    elif p_value <= 0.10:
        zakljucak = "GRANIČNI SLUČAJ (90%)"
        stacionarna = False
    else:
        zakljucak = "NIJE STACIONARNA"
        stacionarna = False

    return {
        "naziv":       naziv,
        "adf_stat":    round(adf_stat, 4),
        "p_value":     round(p_value, 6),
        "n_lags":      result[2],
        "n_obs":       result[3],
        "crit_1pct":   round(crit_vals["1%"],  4),
        "crit_5pct":   round(crit_vals["5%"],  4),
        "crit_10pct":  round(crit_vals["10%"], 4),
        "zakljucak":   zakljucak,
        "stacionarna": stacionarna,
    }


def print_adf_result(result: dict) -> None:
    print(f"{'─' * 50}")
    print(f"  ADF test: {result['naziv']}")
    print(f"{'─' * 50}")
    print(f"  ADF statistika   : {result['adf_stat']}")
    print(f"  p-vrednost       : {result['p_value']}")
    print(f"  Broj lagova      : {result['n_lags']}")
    print(f"  Broj opservacija : {result['n_obs']}")
    print(f"  Kritične vrednosti:")
    print(f"    1%  : {result['crit_1pct']}")
    print(f"    5%  : {result['crit_5pct']}")
    print(f"    10% : {result['crit_10pct']}")
    print(f"  Zaključak: {result['zakljucak']}")
    print(f"{'─' * 50}\n")


def compare_adf(series_dict: dict) -> pd.DataFrame:
    """Pokreće ADF za više serija i vraća DataFrame sa rezultatima."""
    rows = []
    for naziv, series in series_dict.items():
        r = adf_test(series, naziv)
        rows.append({
            "Serija":         r["naziv"],
            "ADF statistika": r["adf_stat"],
            "p-vrednost":     r["p_value"],
            "Krit. 1%":       r["crit_1pct"],
            "Krit. 5%":       r["crit_5pct"],
            "Zaključak":      r["zakljucak"],
        })
    return pd.DataFrame(rows).set_index("Serija")


def plot_acf_pacf(series: pd.Series,
                  lags:  int   = 40,
                  naziv: str   = "Serija",
                  alpha: float = 0.05) -> None:
    """
    ACF i PACF grafici jedan pored drugog.
    
    Kako čitati:
      PACF: nagli pad posle p lagova → AR(p)
      ACF:  nagli pad posle q lagova → MA(q)
      Oba:  postepen pad → mešovita ARMA struktura
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(series.dropna(), lags=lags, alpha=alpha, ax=axes[0],
             color='#2980b9', vlines_kwargs={"colors": "#2980b9"})

    plot_pacf(series.dropna(), lags=lags, alpha=alpha, ax=axes[1],
              method='ywm', color='#e74c3c',
              vlines_kwargs={"colors": "#e74c3c"})

    axes[0].set_title(f'ACF – {naziv}', fontsize=12, fontweight='bold')
    axes[1].set_title(f'PACF – {naziv}', fontsize=12, fontweight='bold')

    for ax in axes:
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Korelacija')

    plt.suptitle(f'ACF i PACF – {naziv}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf_comparison(series_dict: dict, lags: int = 40) -> None:
    """Crta ACF i PACF za više serija – korisno za poređenje pre/posle transformacije."""
    n = len(series_dict)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))

    if n == 1:
        axes = np.array([axes])

    for i, (naziv, series) in enumerate(series_dict.items()):
        plot_acf(series.dropna(), lags=lags, ax=axes[i, 0],
                 color='#2980b9', vlines_kwargs={"colors": "#2980b9"},
                 title=f'ACF – {naziv}')
        plot_pacf(series.dropna(), lags=lags, ax=axes[i, 1],
                  method='ywm', color='#e74c3c',
                  vlines_kwargs={"colors": "#e74c3c"},
                  title=f'PACF – {naziv}')
        for ax in axes[i]:
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_xlabel('Lag')

    plt.suptitle('Poređenje ACF/PACF pre i posle transformacije',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def ljungbox_white_noise_test(series: pd.Series,
                              naziv: str = "Serija",
                              lags:  list = None) -> dict:
    """
    Ljung-Box test na originalnoj seriji prinosa – H0: serija je beli šum.
    
    VAŽNA RAZLIKA: ovo se radi na originalnoj seriji (log-returns),
    NE na rezidualima ARIMA modela! Cilj je da formalno potvrdimo
    (ili odbacimo) hipotezu da su sami prinosi beli šum.
    
    Interpretacija:
      p > 0.05 → ne možemo odbaciti H0 → serija je statistički beli šum
      p ≤ 0.05 → odbacujemo H0 → postoji linearna autokorelacija

    """
    if lags is None:
        n = len(series.dropna())
        # Heuristika: testiramo na log(n), 10, 20 lagovima
        lags = sorted(set([int(np.log(n)), 10, 20, 30]))
        lags = [l for l in lags if l > 0 and l < n // 2]

    series_clean = series.dropna()
    lb = acorr_ljungbox(series_clean, lags=lags, return_df=True)

    print(f"{'─' * 60}")
    print(f"  Ljung-Box test belog šuma – {naziv}")
    print(f"{'─' * 60}")
    print(f"  H0: serija je beli šum (nema linearne autokorelacije)")
    print(f"  N  = {len(series_clean)} opservacija\n")
    print(lb.to_string())

    sve_p = lb['lb_pvalue'].values
    beli_sum = all(p > 0.05 for p in sve_p)
    granicni  = any(0.01 < p <= 0.05 for p in sve_p)

    if beli_sum:
        zakljucak = "NE ODBACUJEMO H0 → serija statistički nije razlikovana od belog šuma"
    elif granicni:
        zakljucak = "GRANIČNI SLUČAJ → neke p-vrednosti su bliske 0.05"
    else:
        zakljucak = "ODBACUJEMO H0 → postoji statistički značajna linearna autokorelacija"

    print(f"\n  Zaključak: {zakljucak}")
    print(f"{'─' * 60}\n")

    return {
        "ljungbox_tabela": lb,
        "beli_sum":        beli_sum,
        "zakljucak":       zakljucak,
        "lags_testirani":  lags,
    }


def runs_test_white_noise(series: pd.Series, naziv: str = "Serija") -> dict:
    """
    Runs (Wald-Wolfowitz) test slučajnosti – H0: serija je slučajna.
    
    Ovaj test je neparametarski i ne oslanja se na normalnost.
    Testira da li su pozitivni [+] i negativni [-] prinosi nasumično raspoređeni.
    Za razliku od Ljung-Box testa (koji testira linearnu autokorelaciju),
    runs test testira opštu slučajnost niza znakova.
    
    """
    series_clean = series.dropna()
    # Klasifikujemo prinos kao pozitivan (1) ili negativan (0)
    znakovi = (series_clean > series_clean.median()).astype(int).values

    n_total    = len(znakovi)
    n_positive = znakovi.sum()
    n_negative = n_total - n_positive

    # Broji runs (uzastopni isti znakovi)
    runs = 1 + np.sum(znakovi[1:] != znakovi[:-1])

    # Z-statistika runs testa
    mu_r    = (2 * n_positive * n_negative) / n_total + 1
    sigma_r = np.sqrt(
        (2 * n_positive * n_negative * (2 * n_positive * n_negative - n_total))
        / (n_total ** 2 * (n_total - 1))
    )
    z_stat = (runs - mu_r) / sigma_r if sigma_r > 0 else 0.0

    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    slucajno = p_value > 0.05
    zakljucak = (
        "NE ODBACUJEMO H0 → niz je slučajan (runs test)"
        if slucajno else
        "ODBACUJEMO H0 → postoji struktura u nizu znakova"
    )

    print(f"{'─' * 60}")
    print(f"  Runs test slučajnosti – {naziv}")
    print(f"{'─' * 60}")
    print(f"  N ukupno       : {n_total}")
    print(f"  N pozitivnih   : {n_positive}")
    print(f"  N negativnih   : {n_negative}")
    print(f"  Broj runs-ova  : {runs:.0f}  (očekivano: {mu_r:.1f})")
    print(f"  Z-statistika   : {z_stat:.4f}")
    print(f"  p-vrednost     : {p_value:.6f}")
    print(f"  Zaključak      : {zakljucak}")
    print(f"{'─' * 60}\n")

    return {
        "runs":       int(runs),
        "z_stat":     round(z_stat, 4),
        "p_value":    round(p_value, 6),
        "slucajno":   slucajno,
        "zakljucak":  zakljucak,
    }


def white_noise_summary(series: pd.Series, naziv: str = "Log-returns") -> None:
    """
    Kompletna analiza belog šuma: Ljung-Box + Runs test + zaključak.
    Ovo je centralna funkcija za tačku 1 ispravke projekta.
    """
    print(f"\n{'═' * 60}")
    print(f"  ANALIZA BELOG ŠUMA: {naziv}")
    print(f"{'═' * 60}\n")

    lb_res   = ljungbox_white_noise_test(series, naziv)
    runs_res = runs_test_white_noise(series, naziv)

    print(f"\n{'═' * 60}")
    print(f"  REKAPITULACIJA:")
    print(f"    Ljung-Box: {lb_res['zakljucak']}")
    print(f"    Runs test: {runs_res['zakljucak']}")

    oba_potvrdjuju = lb_res['beli_sum'] and runs_res['slucajno']
    if oba_potvrdjuju:
        print(f"\n  → Oba testa podržavaju hipotezu o belom šumu.")
        print(f"    Ovo je konzistentno sa Hipotezom efikasnog tržišta (EMH).")
    else:
        print(f"\n  → Testovi daju mešovite ili negativne rezultate.")
        print(f"    Moguće je da postoji neka (slaba) linearna ili nelinearna struktura.")
    print(f"{'═' * 60}\n")


def suggest_arima_params(series: pd.Series,
                         lags:  int   = 40,
                         alpha: float = 0.05) -> dict:
    """
    "Ručna" preporuka za p i q na osnovu ACF/PACF.
    
    Napomena: ovo je samo polazna tačka. Za finalni odabir 
    koristiti auto_arima sa AIC/BIC kriterijumom.
    
    Logika:
      p = poslednji značajni lag u PACF
      q = poslednji značajni lag u ACF
      d = 0 ako je ADF potvrdio stacionarnost, inače 1
    """
    series_clean = series.dropna()
    granica = 1.96 / np.sqrt(len(series_clean))

    # preskačemo lag 0 (uvek je 1.0)
    acf_vals  = acf(series_clean,  nlags=lags, fft=True)[1:]
    pacf_vals = pacf(series_clean, nlags=lags, method='ywm')[1:]

    znacajni_acf  = np.where(np.abs(acf_vals)  > granica)[0]
    znacajni_pacf = np.where(np.abs(pacf_vals) > granica)[0]

    q_predlog = int(znacajni_acf[-1]  + 1) if len(znacajni_acf)  > 0 else 0
    p_predlog = int(znacajni_pacf[-1] + 1) if len(znacajni_pacf) > 0 else 0

    adf_res = adf_test(series_clean)
    d_predlog = 0 if adf_res["stacionarna"] else 1

    return {
        "p": p_predlog,
        "d": d_predlog,
        "q": q_predlog,
        "granica_znacajnosti": round(granica, 4),
        "napomena": "Heuristika – koristiti zajedno sa auto_arima i AIC/BIC."
    }