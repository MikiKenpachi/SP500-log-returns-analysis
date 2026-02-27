"""
pca_analysis.py
---------------
PCA analiza varijansi S&P 500 tržišta na osnovu dnevnih prinosa akcija.

Centralno istraživačko pitanje:
  "Koje kompanije i sektori najviše doprinose varijansi tržišta
   i kako se taj doprinos menja tokom vremena?"

Pristup:
  Ulaz: matrica log-returns (redovi = trading dani, kolone = ~500 akcija)
  PC1  → "tržišni faktor" – sistemski rizik koji pogađa sve akcije
  PC2+ → sektorski/stilski faktori (Growth vs Value, Tech vs Energy...)

  Rolling PCA → kako se struktura varijansi menja tokom kriza
  Loadings    → koje akcije i sektori dominiraju svakim faktorom

Zašto log-returns akcija, a ne finansijski pokazatelji iz companies.csv?
  - Varijansu TRŽIŠTA objašnjavamo kroz zajedničko kretanje CENA
  - Tržišna kapitalizacija i EBITDA su statični – ne pokazuju dinamiku
  - PCA na returns matrici je standardna tehnika u finansijskoj ekonometriji
    (Fama-French faktori, risk parity, portfolio optimization)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Konstante (konzistentno sa preprocessing.py) ─────────────────────────────

TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
# Test: 2024-01-01 → kraj

MIN_POKRIVENOST = 0.85   # akcija mora imati podatke za ≥85% trading dana
ROLLING_WINDOW  = 252    # 1 trading godina za rolling PCA


# ── Učitavanje i priprema ─────────────────────────────────────────────────────

def ucitaj_returns_matricu(stocks_path: str,
                            companies_path: str,
                            start_date: str = None,
                            end_date:   str = None,
                            min_pokrivenost: float = MIN_POKRIVENOST) -> tuple:
    """
    Učitava sp500_stocks.csv i kreira matricu dnevnih log-returns.

    Učitavamo samo kolone Date, Symbol, Adj Close

    Koraci:
      1. Učitaj Date, Symbol, Adj Close
      2. Pivot: redovi = dani, kolone = akcije (Adj Close vrednosti)
      3. Log-returns: ln(P_t / P_{t-1})
      4. Ukloni akcije sa previše NaN (delistirane tokom perioda)
      5. Popuni preostale NaN sa 0 (dan bez promene – konzervativan pristup)

    Parametri:
      stocks_path     : putanja do sp500_stocks.csv
      companies_path  : putanja do sp500_companies.csv (za sektor info)
      start_date      : opciono filtriranje početnog datuma
      end_date        : opciono filtriranje krajnjeg datuma
      min_pokrivenost : minimalni udeo non-NaN vrednosti po akciji

    Vraća:
      returns_mat : pd.DataFrame (dani × akcije) log-returns
      companies   : pd.DataFrame sa Symbol, Sector, Shortname
    """
    print("Učitavamo sp500_stocks.csv (Date, Symbol, Adj Close)...")
    df = pd.read_csv(
        stocks_path,
        usecols=['Date', 'Symbol', 'Adj Close'],
        parse_dates=['Date'],
    )
    print(f"  Učitano: {len(df):,} redova, {df['Symbol'].nunique()} akcija")

    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]

    # Pivot: redovi = dani, kolone = simboli
    prices = df.pivot(index='Date', columns='Symbol', values='Adj Close')
    prices.sort_index(inplace=True)

    # Log-returns
    returns = np.log(prices).diff().iloc[1:]

    # Uklanjamo akcije sa previše NaN
    n_dana = len(returns)
    pokrivenost = returns.notna().sum() / n_dana
    returns = returns.loc[:, pokrivenost >= min_pokrivenost]
    print(f"  Akcija posle filtera : {returns.shape[1]}")

    # Preostale NaN → 0
    returns = returns.fillna(0)

    companies = pd.read_csv(companies_path)[['Symbol', 'Shortname', 'Sector', 'Weight']]

    print(f"  Finalna tabela: {returns.shape[0]} dana × {returns.shape[1]} akcija")
    print(f"  Period: {returns.index[0].date()} → {returns.index[-1].date()}")

    return returns, companies


def podeli_returns(returns: pd.DataFrame) -> tuple:
    """Deli returns matricu na train/val/test konzistentno sa preprocessing.py."""
    train = returns.loc[:TRAIN_END]
    val   = returns.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):VAL_END]
    test  = returns.loc[pd.Timestamp(VAL_END)   + pd.Timedelta(days=1):]

    print(f"Train : {train.shape[0]} dana ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"Val   : {val.shape[0]} dana ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"Test  : {test.shape[0]} dana ({test.index[0].date()} → {test.index[-1].date()})")

    return train, val, test


# ── PCA fitovanje ─────────────────────────────────────────────────────────────

def fituj_pca_returns(returns: pd.DataFrame,
                      n_components: int = 20,
                      standardize: bool = True) -> tuple:
    """
    Fituje PCA na matrici log-returns.

    Napomena o standardizaciji:
      - standardize=True  → svaka akcija dobija isti značaj bez obzira na
                            volatilnost – fokus na KORELACIONU strukturu
      - standardize=False → akcije sa višom volatilnošću dominiraju –
                            fokus na KOVARIJANSU

      Za sektorsku strukturu preporučuje se standardize=True jer ne
      želimo da NVDA automatski dominira samo zbog visoke volatilnosti.

    Vraća:
      pca      : fitovani PCA objekat
      X_pca    : projekcije (factor scores) – kako svaki dan leži na PC osama
      loadings : DataFrame (akcije × komponente) – doprinos svake akcije PC
      scaler   : StandardScaler (ili None)
    """
    X = returns.values

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    print(f"PCA fitovan na {returns.shape[0]} dana × {returns.shape[1]} akcija")
    print(f"Prvih {n_components} PC objašnjava: "
          f"{pca.explained_variance_ratio_[:n_components].sum()*100:.1f}% varijanse")
    print(f"\nVarijansa po PC:")
    kum = 0
    for i, v in enumerate(pca.explained_variance_ratio_[:10]):
        kum += v
        print(f"  PC{i+1}: {v*100:.2f}%  (kumulativno: {kum*100:.1f}%)")

    return pca, X_pca, loadings, scaler


# ── Rolling PCA – vremenski aspekt ───────────────────────────────────────────

def rolling_pca_variance(returns: pd.DataFrame,
                         window: int = ROLLING_WINDOW,
                         n_components: int = 5,
                         step: int = 21) -> pd.DataFrame:
    """
    Rolling PCA: fitujemo PCA na prozoru od `window` dana, pomičemo za `step`.

    Pratimo kako se PC1 explained variance menja tokom vremena.
    Tokom kriza (COVID 2020, 2022) PC1 objašnjava VIŠE varijanse jer
    akcije postaju više korelirane – sistemski rizik dominira.

    Parametri:
      window      : veličina prozora u danima (default: 252 = 1 god.)
      n_components: broj komponenti koje pratimo
      step        : pomak prozora u danima (default: 21 ≈ 1 mesec)

    Vraća:
      df_rolling : DataFrame sa datumima i explained variance za svaki PC
    """
    dates, rezultati = [], []
    n = len(returns)

    print(f"Rolling PCA: window={window} dana, step={step} dana")
    total_steps = (n - window) // step
    print(f"  Ukupno koraka: {total_steps}")

    for start in range(0, n - window, step):
        end = start + window
        prozor = returns.iloc[start:end].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(prozor)

        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_scaled)

        row = {f'PC{i+1}': pca.explained_variance_ratio_[i]
               for i in range(n_components)}
        row['PC_ostatak'] = 1 - pca.explained_variance_ratio_[:n_components].sum()

        dates.append(returns.index[end - 1])
        rezultati.append(row)

        if len(dates) % 20 == 0:
            print(f"  {len(dates)}/{total_steps} koraka...")

    df = pd.DataFrame(rezultati, index=pd.DatetimeIndex(dates))
    print("  Rolling PCA završen.")
    return df


# ── Loadings analiza po sektoru ───────────────────────────────────────────────

def loadings_po_sektoru(loadings: pd.DataFrame,
                         companies: pd.DataFrame,
                         pc: str = 'PC1') -> tuple:
    """
    Spaja loadings sa sektorskom informacijom i agregira po sektoru.

    Za svaki sektor:
      - mean_abs_loading : prosečan apsolutni loading (jačina doprinosa PC)
      - mean_loading     : prosečan loading (smer)
      - n_akcija         : broj akcija

    Odgovara na pitanje: "koji sektori dominiraju datim faktorom?"
    """
    df = loadings[[pc]].copy()
    df.index.name = 'Symbol'
    df = df.reset_index().merge(
        companies[['Symbol', 'Sector', 'Shortname']], on='Symbol', how='left')
    df['Sector'] = df['Sector'].fillna('Ostalo')

    sektor_agg = df.groupby('Sector').agg(
        mean_loading=    (pc, 'mean'),
        mean_abs_loading=(pc, lambda x: x.abs().mean()),
        n_akcija=        (pc, 'count'),
    ).sort_values('mean_abs_loading', ascending=False)

    print(f"\nSektorski doprinos {pc}:")
    print(sektor_agg.round(4).to_string())

    return sektor_agg, df


# ── Vizualizacije ─────────────────────────────────────────────────────────────

def plot_scree(pca: PCA, n_prikaz: int = 20,
               title: str = "S&P 500 returns") -> None:
    """
    Scree plot objašnjene varijanse.
    PC1 u finansijama tipično objašnjava 20-35% – to je 'tržišni faktor'.
    """
    n = min(n_prikaz, len(pca.explained_variance_ratio_))
    var = pca.explained_variance_ratio_[:n] * 100
    kum = np.cumsum(var)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(range(1, n+1), var, color='#3498db', alpha=0.8)
    ax1.set_xlabel('Glavna komponenta - PC1')
    ax1.set_ylabel('Objašnjena varijansa (%)', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')

    ax2 = ax1.twinx()
    ax2.plot(range(1, n+1), kum, 'o--', color='#e74c3c', linewidth=2)
    ax2.axhline(50, color='gray', linestyle=':', linewidth=0.8)
    ax2.set_ylabel('Kumulativna varijansa (%)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 105)

    ax1.set_title(f'Scree plot – PCA na log-returns {title}',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\nPC1 = {var[0]:.1f}% varijanse → tržišni faktor (sistemski rizik)")


def plot_top_loadings(loadings: pd.DataFrame,
                      companies: pd.DataFrame,
                      pc: str = 'PC1',
                      top_n: int = 20) -> None:
    """Top N akcija po apsolutnom loadingu, obojene po sektoru."""
    df = loadings[[pc]].copy()
    df.index.name = 'Symbol'
    df = df.reset_index().merge(
        companies[['Symbol', 'Sector', 'Shortname']], on='Symbol', how='left')
    df = df.reindex(df[pc].abs().sort_values(ascending=False).index)
    df_top = df.head(top_n)

    sektori = df_top['Sector'].fillna('Ostalo').unique()
    boje_mapa = {s: cm.get_cmap('tab20')(i) for i, s in enumerate(sorted(sektori))}
    bar_boje = [boje_mapa[s] for s in df_top['Sector'].fillna('Ostalo')]

    fig, ax = plt.subplots(figsize=(13, max(5, top_n * 0.32)))
    ax.barh(range(len(df_top)), df_top[pc].values,
            color=bar_boje, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(
        [f"{row['Symbol']} – {str(row['Shortname'])[:22]}"
         for _, row in df_top.iterrows()], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'Loading na {pc}')
    ax.set_title(f'Top {top_n} akcija po loadingu na {pc}',
                 fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    legenda = [Patch(facecolor=boje_mapa[s], label=s) for s in sorted(sektori)]
    ax.legend(handles=legenda, loc='lower right', fontsize=7, framealpha=0.8)
    plt.tight_layout()
    plt.show()


def plot_loadings_sektor(sektor_agg: pd.DataFrame, pc: str = 'PC1') -> None:
    """Sektorski prosečni apsolutni loadings."""
    fig, ax = plt.subplots(figsize=(10, 5))
    boje = cm.get_cmap('tab20', len(sektor_agg))
    ax.bar(range(len(sektor_agg)),
           sektor_agg['mean_abs_loading'].values,
           color=[boje(i) for i in range(len(sektor_agg))],
           alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(sektor_agg)))
    ax.set_xticklabels(sektor_agg.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'Prosečni |loading| na {pc}')
    ax.set_title(f'Doprinos sektora faktoru {pc}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rolling_pc1(df_rolling: pd.DataFrame) -> None:
    """
    PC1 explained variance kroz vreme – tržišni faktor.

    Ključna interpretacija:
        PC1 varijansa raste → akcije se kreću zajedno → krizni period (sistemski rizik)
        PC1 varijansa pada  → idiosinkratski rizici dominiraju → mirno tržište
    """
    fig, ax = plt.subplots(figsize=(15, 4))

    ax.plot(df_rolling.index, df_rolling['PC1'] * 100,
            color='#e74c3c', linewidth=1.2)
    ax.fill_between(df_rolling.index, df_rolling['PC1'] * 100,
                    alpha=0.15, color='#e74c3c')

    mean_pc1 = df_rolling['PC1'].mean() * 100
    ax.axhline(mean_pc1, color='gray', linestyle='--', linewidth=0.8,
               label=f'Prosek: {mean_pc1:.1f}%')

    eventi = {
        '2018-12-24': 'Pad tržišta krajem 2018',
        '2020-03-23': 'COVID',
        '2022-01-01': 'Monetarna kriza SAD',
    }
    ymax = df_rolling['PC1'].max() * 100
    for datum_str, naziv in eventi.items():
        datum = pd.Timestamp(datum_str)
        if df_rolling.index[0] <= datum <= df_rolling.index[-1]:
            ax.axvline(datum, color='#2c3e50', linestyle=':', linewidth=1.2)
            ax.text(datum, ymax * 0.95, naziv, rotation=90,
                    fontsize=7, va='top', ha='right', color='#2c3e50')

    ax.set_title('PC1 varijansa kroz vreme – tržišni faktor (sistemski rizik)\n'
                 'Viša vrednost = akcije se kreću zajedno = veći sistemski rizik',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('PC1 varijansa (%)')
    ax.set_xlabel('Datum')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_rolling_stacked(df_rolling: pd.DataFrame, n_pc: int = 5) -> None:
    """
    Stacked area chart svih PC komponenti kroz vreme.
    Prikazuje kako se varijansa komponenti menja tokom različitih tržišnih perioda.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    boje = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#bdc3c7']
    pc_cols = [c for c in [f'PC{i+1}' for i in range(n_pc)]
               if c in df_rolling.columns]

    vrednosti = [df_rolling[pc].values for pc in pc_cols]
    if 'PC_ostatak' in df_rolling.columns:
        vrednosti.append(df_rolling['PC_ostatak'].values)
        labele = pc_cols + ['Ostatak']
        boje_plot = boje[:len(pc_cols)] + ['#bdc3c7']
    else:
        labele = pc_cols
        boje_plot = boje[:len(pc_cols)]

    ax.stackplot(df_rolling.index, vrednosti,
                 labels=labele, colors=boje_plot, alpha=0.75)

    eventi = {'2020-03-23': 'COVID', '2022-01-01': 'Monetarna kriza u SAD'}
    for datum_str, naziv in eventi.items():
        datum = pd.Timestamp(datum_str)
        if df_rolling.index[0] <= datum <= df_rolling.index[-1]:
            ax.axvline(datum, color='black', linestyle='--',
                       linewidth=1.2, alpha=0.6)
            ax.text(datum, 1.01, naziv, fontsize=8, ha='center',
                    transform=ax.get_xaxis_transform())

    ax.set_title('Rolling PCA – struktura varijansi tržišta kroz vreme\n'
                 f'(prozor: {ROLLING_WINDOW} trading dana)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Udeo objašnjene varijanse')
    ax.set_xlabel('Datum')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_akcije_pc_prostor(loadings: pd.DataFrame,
                            companies: pd.DataFrame,
                            pc_x: str = 'PC1',
                            pc_y: str = 'PC2') -> None:
    """
    Scatter akcija u PC1-PC2 prostoru, obojeno po sektoru.
    Akcije blizu u ovom prostoru reaguju slično na iste tržišne faktore.
    """
    df = loadings[[pc_x, pc_y]].copy()
    df.index.name = 'Symbol'
    df = df.reset_index().merge(
        companies[['Symbol', 'Sector']], on='Symbol', how='left')
    df['Sector'] = df['Sector'].fillna('Ostalo')

    sektori = sorted(df['Sector'].unique())
    boje_mapa = {s: cm.get_cmap('tab20')(i) for i, s in enumerate(sektori)}

    fig, ax = plt.subplots(figsize=(13, 9))
    for sektor in sektori:
        maska = df['Sector'] == sektor
        ax.scatter(df.loc[maska, pc_x], df.loc[maska, pc_y],
                   color=boje_mapa[sektor], label=sektor,
                   alpha=0.55, s=25, edgecolors='none')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(f'{pc_x} loading', fontsize=11)
    ax.set_ylabel(f'{pc_y} loading', fontsize=11)
    ax.set_title(f'Akcije u {pc_x}-{pc_y} prostoru (obojeno po sektoru)\n'
                 'Bliske akcije reaguju slično na iste tržišne faktore',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2,
              markerscale=2, framealpha=0.8)
    plt.tight_layout()
    plt.show()


def plot_sektorski_pc_prostor(loadings: pd.DataFrame,
                               companies: pd.DataFrame,
                               pca: PCA) -> None:
    """
    Prosečna pozicija sektora u PC1-PC2 prostoru loadings-a.
    Veličina balona ∝ sqrt(tržišne težine) – sqrt skala za čitljivost.
    """
    df = loadings[['PC1', 'PC2']].copy()
    df.index.name = 'Symbol'
    df = df.reset_index().merge(
        companies[['Symbol', 'Sector', 'Weight']], on='Symbol', how='left')

    sektor_agg = df.groupby('Sector').agg(
        PC1=('PC1', 'mean'),
        PC2=('PC2', 'mean'),
        N=('PC1', 'count'),
        Weight=('Weight', 'sum'),
    ).reset_index()

    raw = sektor_agg['Weight'].fillna(sektor_agg['N'] / 500)
    sizes = (np.sqrt(raw / raw.max()) * 3000).clip(lower=80)

    fig, ax = plt.subplots(figsize=(13, 8))
    boje = cm.get_cmap('tab20', len(sektor_agg))

    for i, row in sektor_agg.iterrows():
        ax.scatter(row['PC1'], row['PC2'],
                   s=float(sizes.iloc[i]),
                   color=boje(i), alpha=0.8,
                   edgecolors='white', linewidths=1.5)
        ax.annotate(row['Sector'],
                    xy=(row['PC1'], row['PC2']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color='#2c3e50')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(f"PC1 loading ({pca.explained_variance_ratio_[0]*100:.1f}% var.)",
                  fontsize=11)
    ax.set_ylabel(f"PC2 loading ({pca.explained_variance_ratio_[1]*100:.1f}% var.)",
                  fontsize=11)
    ax.set_title('Prosečna pozicija sektora u PC1-PC2 prostoru\n'
                 '(veličina kružnice - skalirana)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ── Interpretacija ────────────────────────────────────────────────────────────

def interpretiraj_faktore(loadings: pd.DataFrame,
                           companies: pd.DataFrame,
                           n_pc: int = 4,
                           top_n: int = 5) -> None:
    """
    Tekstualna interpretacija finansijskih faktora na osnovu top loadings.

    PC1 → "tržišni faktor": sve akcije imaju pozitivan loading
          (sistematski se kreću u istom smeru, samo različitim intenzitetom)
    PC2 → kontrastni faktor: neki sektori + neki - (sektorski kontrasti)
    """
    df = loadings.copy()
    df.index.name = 'Symbol'
    df = df.reset_index().merge(
        companies[['Symbol', 'Sector']], on='Symbol', how='left')

    print("=" * 60)
    print("  INTERPRETACIJA FINANSIJSKIH FAKTORA")
    print("=" * 60)

    for i in range(min(n_pc, loadings.shape[1])):
        pc = f'PC{i+1}'
        ev = loadings[pc].var()
        print(f"\n{pc}:")

        top_pos = df.nlargest(top_n, pc)[['Symbol', 'Sector', pc]]
        top_neg = df.nsmallest(top_n, pc)[['Symbol', 'Sector', pc]]

        print(f"  Najveći + loadings (akcije koje RASTU sa ovim faktorom):")
        for _, row in top_pos.iterrows():
            print(f"    +{row[pc]:.4f}  {row['Symbol']:6s}  ({row['Sector']})")

        print(f"  Najveći - loadings (akcije koje PADAJU sa ovim faktorom):")
        for _, row in top_neg.iterrows():
            print(f"     {row[pc]:.4f}  {row['Symbol']:6s}  ({row['Sector']})")

        pos_sektori = top_pos['Sector'].value_counts().index[:2].tolist()
        neg_sektori = top_neg['Sector'].value_counts().index[:2].tolist()

        if i == 0:
            print(f"\n  → PC1 = TRŽIŠNI FAKTOR: sve akcije reaguju zajedno "
                  f"(sistemski rizik)")
        else:
            print(f"\n  → Kontrast: {', '.join(pos_sektori)} "
                  f"vs {', '.join(neg_sektori)}")

    print("=" * 60)
