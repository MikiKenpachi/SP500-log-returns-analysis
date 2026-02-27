📈 **Analiza i predikcija kratkoročnih logaritamskih prinosa S&P 500 indeksa**

Ovaj projekat predstavlja sveobuhvatnu analizu i modelovanje vremenskih serija indeksa **S&P 500**, sa fokusom na **logaritamske prinose (log-returns)** umesto direktnog predviđanja cena.

Log-prinosi su statistički pogodniji za modelovanje jer su bliži stacionarnosti i omogućavaju pouzdaniju procenu tržišnog rizika.

**Autor**: Miloš Trišić (RA39/2023)<br>
**Predmet**: Numerički algoritmi i numerički softver (NANS)<br>
**Fakultet**: Fakultet tehničkih nauka<br>

📊 **Pregled projekta**

Cilj rada je evaluacija različitih pristupa modelovanju i predikciji finansijskih kretanja:

**ARIMA** – linearno modelovanje i analiza autokorelacione strukture

**Facebook Prophet** – identifikacija trenda i sezonalnosti

**GARCH** – modelovanje volatilnosti i procena tržišnog rizika

**PCA analiza** – razumevanje strukture tržišta i sektorskih doprinosa varijansi

🛠 **Tehnologije i biblioteke**

**Python 3.x**

**Pandas**, **NumPy** – obrada i manipulacija podataka

**Statsmodels**, **pmdarima** – ARIMA modeli

**Prophet** – nelinearno modelovanje

**arch** – GARCH modeli

**Scikit-learn** – PCA analiza

**Matplotlib, Seaborn** – vizuelizacija

📂 **Struktura projekta**

Projekat je organizovan modularno radi preglednosti i lakšeg održavanja:

notebook.ipynb – kompletan tok analize i interpretacija rezultata

preprocessing.py – učitavanje podataka i izračunavanje log-prinosa

stationarity.py – ADF test i ACF/PACF analiza

arima_model.py – ARIMA sa walk-forward validacijom

prophet_model.py – implementacija Prophet modela

garch_model.py – modelovanje volatilnosti

pca_analysis.py – analiza glavnih komponenti

evaluation.py – centralizovane metrike (MAE, RMSE, MASE)

📈 **Ključni rezultati i zaključci**

**Efikasnost tržišta**: Log-prinosi su veoma bliski belom šumu, što ograničava linearnu predikciju.

**Volatilnost**: Uočeno je klasterovanje volatilnosti, uspešno modelovano GARCH(1,1) modelom.

**Walk-forward validacija**: Simulira realne uslove i eliminiše look-ahead bias.

**MASE metrika**: Omogućava poređenje modela sa naivnim pristupom.

**PCA analiza**: Prva glavna komponenta (PC1) predstavlja opšti tržišni faktor, dok PC2 razdvaja defanzivne i ciklične sektore.

🚀 **Pokretanje projekta**

Kloniranje repozitorijuma:

git clone https://github.com/vas-username/sp500-analysis.git
cd sp500-analysis

Instalacija biblioteka:

pip install pandas numpy statsmodels pmdarima prophet arch scikit-learn matplotlib seaborn

Pokretanje analize:

jupyter notebook
📝 Licenca

Projekat je realizovan u svrhe akademskog istraživanja na Fakultetu tehničkih nauka.
