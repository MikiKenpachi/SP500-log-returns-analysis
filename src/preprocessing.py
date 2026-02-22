"""
preprocessing.py
----------------
Učitavanje podataka, izračunavanje log-returns i podela na
trening / validacioni / test skup po vremenskom redosledu.
"""

import pandas as pd
import numpy as np


# ── Konstante ────────────────────────────────────────────────────────────────

# Granice vremenskog splita (strogo hronološki, bez preskakanja)
TRAIN_END   = "2022-12-31"
VAL_END     = "2023-12-31"
# Test skup: 2024-01-01 → kraj dataseta (2024-12-20)


# ── Učitavanje i osnovno čišćenje ────────────────────────────────────────────

def load_index(filepath: str) -> pd.DataFrame:
    """
    Učitavamo sp500_index.csv i vraćamo DataFrame sa:
      - DatetimeIndex (Date)
      - kolonom 'S&P500' (float)
      - kolonom 'log_return' (float): ln(P_t / P_{t-1})

    Nedostajući datumi (vikendi, praznici) su prirodno odsutni –
    ne vrši se reindeksiranje jer modeli rade na trading-day frekvenciji.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date",)

    # Proveravamo da li postoje duplikati datuma
    if df.index.duplicated().any():
        raise ValueError("Duplikati datuma pronađeni u index fajlu.")

    df.sort_index(inplace=True)

    # Log-returns: ln(P_t) - ln(P_{t-1})
    # Prva vrednost će biti NaN – uklanjamo je
    df["log_return"] = np.log(df["S&P500"]).diff()
    df.dropna(subset=["log_return"], inplace=True)

    return df


def load_companies(filepath: str) -> pd.DataFrame:
    """
    Učitava sp500_companies.csv.
    Vraća DataFrame sa svim kolonama.
    """
    df = pd.read_csv(filepath)
    return df


# ── Podela na skupove ────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deli DataFrame na trening, validacioni i test skup
    strogo po vremenskom redosledu.

    Povratne vrednosti:
        train  – 2014-12-23 → 2022-12-31
        val    – 2023-01-01 → 2023-12-31
        test   – 2024-01-01 → 2024-12-20
    """
    train = df.loc[:TRAIN_END].copy()
    val   = df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1) : VAL_END].copy()
    test  = df.loc[pd.Timestamp(VAL_END)   + pd.Timedelta(days=1) :].copy()

    # Sanity check – skupovi ne smeju da se preklapaju
    assert train.index.max() < val.index.min(), "Train i val se preklapaju!"
    assert val.index.max()   < test.index.min(), "Val i test se preklapaju!"

    return train, val, test


# ── Pomoćna funkcija: info o podeli ─────────────────────────────────────────

def print_split_info(train: pd.DataFrame,
                     val:   pd.DataFrame,
                     test:  pd.DataFrame) -> None:
    """Štampa pregled podele podataka."""
    total = len(train) + len(val) + len(test)

    for naziv, skup in [("Train", train), ("Validation", val), ("Test", test)]:
        print(
            f"{naziv:>12}: {skup.index.min().date()} → {skup.index.max().date()} "
            f"| {len(skup):>4} trading dana "
            f"({100 * len(skup) / total:.1f}%)"
        )
    print(f"{'Ukupno':>12}: {total} trading dana")


# ── Glavna funkcija: sve u jednom pozivu ─────────────────────────────────────

def load_and_prepare(index_path: str,
                     companies_path: str
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Poziva sve korake redom i vraća:
        train, val, test  – podeljeni index DataFrame-ovi
        companies         – DataFrame kompanija

    Primer upotrebe iz notebook-a:
        from src.preprocessing import load_and_prepare
        train, val, test, companies = load_and_prepare(
            "data/sp500_index.csv",
            "data/sp500_companies.csv"
        )
    """
    index_df  = load_index(index_path)
    companies = load_companies(companies_path)
    train, val, test = split_data(index_df)

    return train, val, test, companies