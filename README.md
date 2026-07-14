# Predicting Bitcoin Price Direction

**ADS 504 — Final Team Project**
University of San Diego · M.S. Applied Data Science

**Team:** _add member names here_

---

## Problem definition

**What is the problem we are solving?**
We frame next-day Bitcoin movement as a **binary classification** problem: given market
data up to and including today, will BTC's closing price be **higher (1)** or **lower (0)**
in the next period? The target is the *direction* of the move, not the price itself.

**Why does it need to be solved?**
Bitcoin is highly volatile and traded around the clock, and directional forecasts feed
directly into trading, hedging, and risk-management decisions. A model that beats a naive
baseline (e.g., "always predict up," or the majority class) — even by a small, consistent
margin — is both practically useful and a clean testbed for the classification and
feature-engineering methods covered in this course.

**What does the machine-learning model solve?**
The model learns the relationship between engineered market signals (momentum, trend,
volatility, volume) and the *sign* of the next return. It turns a large, noisy stream of
OHLCV data into a single actionable probability of an up-move, and lets us compare how
different algorithm families capture (or fail to capture) that signal.

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

> ⚠️ **Leakage warning:** the target is the *future* move. Compute every feature using only
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

**Validation:** a chronological (no-shuffle) train/test split plus
`TimeSeriesSplit` cross-validation. **Metrics:** accuracy, precision, recall, F1,
ROC-AUC, and a confusion matrix — reported against a majority-class baseline so
"better than guessing" is explicit.

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

---

## Workflow

1. **`01_eda_preprocessing`** — load the raw klines, parse timestamps, check for
   missing/duplicate rows and outliers, and profile the series.
2. **`02_feature_engineering`** — build the technical indicators and the `direction`
   target; select/rank candidate features.
3. **`03_modeling`** — chronological split, cross-validate, train all four models, and
   assemble the comparison table + plots that go into the report.
