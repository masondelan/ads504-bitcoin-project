# Predicting Bitcoin Price Direction

**ADS 504 — Final Team Project**
University of San Diego · M.S. Applied Data Science

**Team:** Mehson Mason Delan, Nikita Rogers, Dana Neuman

---

## Problem definition

**What is the problem we are solving?**
We want to predict whether Bitcoin's price will go **up or down** the next period. Instead
of predicting the exact price, we predict the direction — up (**1**) or down (**0**). This
makes it a **binary classification** problem, using only market data available up to and
including the current period.

**Why does it need to be solved?**
Bitcoin's price moves a lot, and traders and investors make decisions based on where it's
likely headed. A model that predicts direction even slightly better than chance is useful —
and it's a strong way to practice the classification and feature-engineering methods from
this course.

**What does the machine-learning model solve?**
The model looks at past price patterns like momentum, trend, and volatility — and learns
which ones tend to come before an up period or a down period. We train several models and
compare which one predicts direction best.

---

## Dataset

**Source:** Binance public market data — <https://github.com/binance/binance-public-data>
(bulk CSV archives served at <https://data.binance.vision>).

We use **BTCUSDT klines** (candlesticks). Using **1-hour** candles over a few years yields
tens of thousands of rows, comfortably clearing the course's **10,000-row minimum**
(daily candles alone would not — hourly or finer is the safe choice).

Each kline CSV row has these columns (no header in the raw files):

| # | Column | Notes |
|---|--------|-------|
| 0 | Open time | ms epoch — parse to datetime |
| 1 | Open | |
| 2 | High | |
| 3 | Low | |
| 4 | Close | |
| 5 | Volume | base-asset volume |
| 6 | Close time | ms epoch |
| 7 | Quote asset volume | |
| 8 | Number of trades | |
| 9 | Taker buy base volume | |
| 10 | Taker buy quote volume | |
| 11 | Ignore | drop |

### Target

`direction = 1 if next_close > current_close else 0` (shift the close by one period).
Check the class balance — it is usually close to 50/50 but rarely exactly, which matters
for choosing metrics and baselines.

### Candidate features

Raw OHLCV plus engineered technical indicators:

- **Trend:** simple / exponential moving averages (SMA, EMA) and their crossovers
- **Momentum:** RSI, MACD (line, signal, histogram)
- **Volatility:** Bollinger Bands (upper/lower/width), rolling std
- **Returns / volume:** log returns, lagged returns, volume change

> **Leakage warning:** the target is the *future* move. Compute every feature using only
> information available *at or before* the current candle, and never shuffle the rows when
> splitting — this is time-series data.

---

## Models

| Model | Family | Library |
|-------|--------|---------|
| Logistic Regression | linear baseline | scikit-learn |
| Random Forest | bagged trees | scikit-learn |
| XGBoost | gradient boosting | `xgboost` |
| LightGBM | gradient boosting | `lightgbm` |

---

## Repository structure

```
ads504-bitcoin-project/
├── data/                     # local market data (git-ignored — see "Getting the data")
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb    # load, clean, explore, handle outliers/missing
│   ├── 02_feature_engineering.ipynb  # indicators + target construction
│   └── 03_modeling.ipynb             # train, validate, compare the 4 models
├── src/                      # optional: shared helper code (data loading, indicators)
├── reports/figures/          # exported plots for the report / slides
├── requirements.txt
└── README.md
```

---

## Getting the data

Raw market data is **not** committed (it's large and reproducible). Download it locally:

**Option A — official scripts.** Clone <https://github.com/binance/binance-public-data>
and run its Python download utilities for `BTCUSDT`.

**Option B — direct download.** Monthly archives follow this URL pattern:

```
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-YYYY-MM.zip
```

Download the months you need, unzip the CSVs into `data/`, and load them in
`01_eda_preprocessing.ipynb`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                       # or open the notebooks in Google Colab
```
## Technologies

- Python
- Google Colab
- GitHub
- Microsoft Word
- PowerPoint
- Slack
- ChatGPT 5.4
- Google Gemini
