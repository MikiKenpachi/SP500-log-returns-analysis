Analiza i predikcija kratkoročnih logaritamskih prinosa S&P 500 indeksa
Ovaj projekat predstavlja sveobuhvatnu analizu i modelovanje vremenskih serija S&P 500 indeksa. Fokus nije na direktnom predviđanju originalnih cena, već na logaritamskim prinosima (log-returns), koji su statistički stabilniji za modelovanje i ključni za procenu rizika u finansijama.

Autor: Miloš Trišić (RA39/2023)

Projekat iz predmeta: Numerički algoritmi i numerički softver (NANS)

📊 Pregled projekta
Cilj rada je evaluacija različitih pristupa predikciji finansijskih kretanja:

Linearno modelovanje (ARIMA) za identifikaciju autokorelacione strukture.

Nelinearno modelovanje (Facebook Prophet) za prepoznavanje sezonalnosti.

Modelovanje volatilnosti (GARCH) radi procene tržišnog rizika.

Deskriptivna PCA analiza za razumevanje strukture tržišta i doprinosa različitih sektora ukupnoj varijansi.

🛠 Tehnologije i biblioteke
Jezik: Python 3.x

Analiza podataka: Pandas, NumPy

Vremenske serije: Statsmodels, pmdarima, Prophet, arch (GARCH modeli)

Mašinsko učenje: Scikit-learn (PCA analiza)

Vizuelizacija: Matplotlib, Seaborn

📂 Struktura projekta
Projekat je organizovan modularno radi lakšeg održavanja i testiranja:

notebook.ipynb — Glavni Jupyter Notebook sa celokupnim tokom analize i interpretacijom rezultata.

preprocessing.py — Učitavanje podataka, izračunavanje log-returns i hronološka podela na train/val/test skupove.

stationarity.py — ADF testovi i analiza ACF/PACF funkcija.

arima_model.py — Implementacija ARIMA modela sa walk-forward validacijom.

prophet_model.py — Implementacija Facebook Prophet modela.

garch_model.py — Modelovanje volatilnosti na rezidualima ARIMA modela.

pca_analysis.py — Analiza glavnih komponenti nad prinosima pojedinačnih akcija.

evaluation.py — Centralizovane metrike (MAE, RMSE, MASE).

📈 Ključni rezultati i zaključci
Efikasnost tržišta: Potvrđeno je da su log-returns veoma bliski "belom šumu", što otežava linearnu predikciju, ali omogućava precizno modelovanje rizika kroz GARCH.

Walk-Forward Validacija: Modeli su testirani simulacijom realnog trgovanja, gde se model konstantno ažurira novim podacima.

MASE Metrika: Korišćena je Mean Absolute Scaled Error kako bi se utvrdilo da li su modeli zaista bolji od najjednostavnijeg "naivnog" pogađanja.

PCA uvid: Analiza je pokazala da PC1 (prva glavna komponenta) predstavlja opšti tržišni rizik, dok PC2 jasno razdvaja defanzivne sektore (Utilities) od cikličnih (Energy, Industrials).

🚀 Kako pokrenuti projekat
Klonirajte repozitorijum:

git clone https://github.com/vas-username/sp500-analysis.git
Instalirajte potrebne biblioteke:

pip install pandas numpy statsmodels pmdarima prophet arch scikit-learn matplotlib seaborn
Pokrenite notebook.ipynb kroz Jupyter ili VS Code.

📝 Licenca
Ovaj projekat je urađen u svrhe akademskog istraživanja na Fakultetu tehničkih nauka.
