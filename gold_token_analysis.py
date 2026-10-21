"""
Tokenised Gold: Tracking Accuracy and Mean-Reversion Analysis
=============================================================

PAXG (Paxos Gold) and XAUT (Tether Gold) are crypto tokens that are each
backed by one troy ounce of physical gold. In theory, each should always
trade at the price of gold, and therefore at the same price as each other.

This script answers three questions:
  1. How accurately does each token track the gold price?
  2. When PAXG and XAUT drift apart, how quickly does the gap close?
     (measured as a half-life, using an Ornstein-Uhlenbeck / AR(1) model)
  3. Could you profit from that gap closing, after transaction costs?
     (a z-score mean-reversion strategy, i.e. simple statistical arbitrage)

How to run:
  pip install yfinance pandas numpy statsmodels matplotlib
  python gold_token_analysis.py

Output: printed results, charts in ./output/ and an auto-generated README.md.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)  # hide library notices


# ---------------------------------------------------------------------------
# SETTINGS: change these to experiment
# ---------------------------------------------------------------------------
START_DATE = "2020-01-01"   # XAUT launched in early 2020
END_DATE = None             # None = up to today

TICKERS = {
    "PAXG": "PAXG-USD",     # Paxos Gold token, priced in US dollars
    "XAUT": "XAUT-USD",     # Tether Gold token, priced in US dollars
    "GOLD": "GC=F",         # COMEX gold futures (front month), our gold benchmark
}

Z_WINDOW = 30         # days of history used to judge whether today's gap is "unusual"
ENTRY_Z = 2.0         # open a trade when the gap is more than 2 standard deviations wide
EXIT_Z = 0.0          # close the trade when the gap returns to its recent average
FEE_PER_LEG = 0.001   # 0.10% fee each time we buy or sell one token

CRYPTO_DAYS = 365     # crypto trades every day of the year
STOCK_DAYS = 252      # gold futures trade on roughly 252 weekdays a year

OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# STEP 1: DOWNLOAD THE DATA
# ---------------------------------------------------------------------------
def download_close(ticker: str) -> pd.Series:
    """Download daily closing prices for one ticker from Yahoo Finance."""
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker "
                           f"or your internet connection.")

    close = df["Close"]
    # Newer yfinance versions return a one-column table instead of a series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Make the dates plain calendar dates so different assets line up
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    close.index = idx.normalize()

    return close.dropna().astype(float)


def load_prices() -> dict:
    """Download every ticker and return a dict of price series."""
    prices = {}
    for name, ticker in TICKERS.items():
        prices[name] = download_close(ticker)
        print(f"  {name:<5} {ticker:<9} {len(prices[name]):>5} days  "
              f"({prices[name].index[0].date()} to {prices[name].index[-1].date()})")
    return prices


# ---------------------------------------------------------------------------
# STEP 2: HOW WELL DOES EACH TOKEN TRACK GOLD?
# ---------------------------------------------------------------------------
def tracking_vs_gold(token: pd.Series, gold: pd.Series, name: str) -> tuple:
    """
    Compare a token with the gold benchmark.

    Premium:        how much more (or less) the token costs than gold, in %.
    Tracking error: how much the token's daily returns differ from gold's,
                    scaled to a yearly figure. Lower = tracks gold better.
    Beta and R²:    from regressing token returns on gold returns.
                    Beta near 1 and a high R² mean the token moves with gold.
    """
    # Keep only days where both have a price (gold futures don't trade at weekends)
    both = pd.concat([token.rename(name), gold.rename("GOLD")],
                     axis=1, join="inner").dropna()

    premium = both[name] / both["GOLD"] - 1

    # Log returns: the % change from one available day to the next
    returns = np.log(both).diff().dropna()
    diff = returns[name] - returns["GOLD"]
    tracking_error = diff.std() * np.sqrt(STOCK_DAYS)

    # Regression: token return = alpha + beta * gold return
    X = sm.add_constant(returns["GOLD"])
    model = sm.OLS(returns[name], X).fit()

    stats = {
        "days compared": len(both),
        "mean premium": premium.mean(),
        "premium st. dev.": premium.std(),
        "days >0.5% off gold": (premium.abs() > 0.005).mean(),
        "tracking error (annual)": tracking_error,
        "beta": model.params["GOLD"],
        "R squared": model.rsquared,
    }
    return stats, premium


# ---------------------------------------------------------------------------
# STEP 3 + 4: THE PAXG vs XAUT SPREAD, AND HOW FAST IT MEAN-REVERTS
# ---------------------------------------------------------------------------
def build_spread(paxg: pd.Series, xaut: pd.Series) -> pd.Series:
    """
    Spread = ln(PAXG) - ln(XAUT), which is roughly the % gap between them.
    Both tokens trade 24/7 on the same markets, so the dates line up exactly.
    """
    both = pd.concat([paxg.rename("PAXG"), xaut.rename("XAUT")],
                     axis=1, join="inner").dropna()
    return (np.log(both["PAXG"]) - np.log(both["XAUT"])).rename("spread")


def mean_reversion_stats(spread: pd.Series) -> dict:
    """
    ADF test: does the spread keep pulling back to an average (stationary),
              or does it wander off randomly? p-value < 0.05 = mean-reverting.

    Half-life: regress today's change in the spread on yesterday's level:
                   change_t = a + b * spread_(t-1) + noise
               If b is negative, gaps get pulled back towards the average.
               This is the discrete-time version of an Ornstein-Uhlenbeck
               process dX = theta * (mu - X) dt + sigma dW, with
               theta = -ln(1 + b) per day.
               Half-life = ln(2) / theta = the days for a gap to halve.
    """
    adf_stat, adf_p = adfuller(spread.dropna())[:2]

    data = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
    data.columns = ["change", "lagged_level"]
    ar = sm.OLS(data["change"], sm.add_constant(data["lagged_level"])).fit()
    b = ar.params["lagged_level"]

    if -1 < b < 0:
        theta = -np.log(1 + b)
        half_life = np.log(2) / theta
    else:
        theta, half_life = np.nan, np.nan  # no mean reversion detected

    return {
        "mean spread": spread.mean(),
        "spread st. dev.": spread.std(),
        "ADF statistic": adf_stat,
        "ADF p-value": adf_p,
        "AR(1) coefficient b": b,
        "OU speed theta (per day)": theta,
        "half-life (days)": half_life,
    }


# ---------------------------------------------------------------------------
# STEP 5: THE TRADING STRATEGY
# ---------------------------------------------------------------------------
def run_strategy(spread: pd.Series) -> pd.DataFrame:
    """
    z-score = (spread - 30-day average) / 30-day standard deviation
              i.e. how unusual today's gap is compared with the last month.

    Position:  +1 = long the spread  = buy PAXG and short XAUT (PAXG is cheap)
               -1 = short the spread = short PAXG and buy XAUT (PAXG is expensive)
                0 = no trade

    Rules:     z < -2 -> go long the spread;  z > +2 -> go short the spread
               exit when z crosses back through 0.

    No look-ahead: a decision made on today's close is only applied from
    tomorrow (shift by one day), so we never trade on information we wouldn't
    have had at the time.

    Returns are measured per £1 put into each leg.
    """
    rolling_mean = spread.rolling(Z_WINDOW).mean()
    rolling_std = spread.rolling(Z_WINDOW).std()
    z = (spread - rolling_mean) / rolling_std

    # Walk through the days one at a time, remembering whether we're in a trade
    signal = np.zeros(len(z))
    current = 0
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            signal[i] = current
            continue
        if current == 0:
            if zi > ENTRY_Z:
                current = -1
            elif zi < -ENTRY_Z:
                current = 1
        elif current == 1 and zi >= EXIT_Z:
            current = 0
        elif current == -1 and zi <= EXIT_Z:
            current = 0
        signal[i] = current

    signal = pd.Series(signal, index=z.index)
    position = signal.shift(1).fillna(0)           # act the day after the signal

    gross = position * spread.diff().fillna(0)     # profit from the gap moving
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * 2 * FEE_PER_LEG             # two tokens traded each time
    net = gross - costs

    return pd.DataFrame({
        "spread": spread, "z": z, "signal": signal, "position": position,
        "gross": gross, "costs": costs, "net": net,
        "equity": np.exp(net.cumsum()),
        "equity_gross": np.exp(gross.cumsum()),
    })


def performance(daily_returns: pd.Series, equity: pd.Series,
                position: pd.Series) -> dict:
    """Standard performance metrics for a daily return series."""
    years = len(daily_returns) / CRYPTO_DAYS
    std = daily_returns.std()
    drawdown = equity / equity.cummax() - 1
    entries = ((position != 0) & (position.shift(1).fillna(0) == 0)).sum()

    return {
        "total return": equity.iloc[-1] - 1,
        "annual return": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "annual volatility": std * np.sqrt(CRYPTO_DAYS),
        "Sharpe ratio": (daily_returns.mean() / std * np.sqrt(CRYPTO_DAYS)
                         if std > 0 else np.nan),
        "max drawdown": drawdown.min(),
        "number of trades": int(entries),
        "% of days in a trade": (position != 0).mean(),
    }


# ---------------------------------------------------------------------------
# STEP 6: CHARTS AND REPORT
# ---------------------------------------------------------------------------
def make_charts(premiums: dict, bt: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Chart 1: each token's premium/discount versus gold
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, prem in premiums.items():
        ax.plot(prem.index, prem * 100, label=name, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Token premium vs gold futures (%)")
    ax.set_ylabel("%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_premium_vs_gold.png", dpi=150)
    plt.close(fig)

    # Chart 2: PAXG-XAUT spread with trade entries marked
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(bt.index, bt["spread"] * 10_000, color="grey", linewidth=0.8,
            label="spread")
    entries = bt[(bt["signal"] != 0) & (bt["signal"].shift(1).fillna(0) == 0)]
    longs = entries[entries["signal"] == 1]
    shorts = entries[entries["signal"] == -1]
    ax.scatter(longs.index, longs["spread"] * 10_000, color="green", s=14,
               label="long spread (buy PAXG, short XAUT)")
    ax.scatter(shorts.index, shorts["spread"] * 10_000, color="red", s=14,
               label="short spread (short PAXG, buy XAUT)")
    ax.set_title("PAXG - XAUT spread (basis points) and trade entries")
    ax.set_ylabel("bps")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_spread_and_trades.png", dpi=150)
    plt.close(fig)

    # Chart 3: equity curve, before and after fees
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(bt.index, bt["equity_gross"], label="before fees", linewidth=1)
    ax.plot(bt.index, bt["equity"], label="after fees", linewidth=1.4)
    ax.axhline(1, color="black", linewidth=0.6)
    ax.set_title("Mean-reversion strategy: growth of £1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_equity_curve.png", dpi=150)
    plt.close(fig)


def fmt(key: str, value) -> str:
    """Format a number sensibly for printing."""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    pct_words = ("premium", "return", "volatility", "drawdown", "%", "error",
                 "spread", "off gold")
    if any(w in key for w in pct_words) and "days compared" not in key:
        return f"{value * 100:.3f}%"
    return f"{value:.4f}"


def print_table(title: str, stats: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in stats.items():
        print(f"  {k:<28} {fmt(k, v)}")


def md_table(stats: dict) -> str:
    rows = ["| Metric | Value |", "|---|---|"]
    rows += [f"| {k} | {fmt(k, v)} |" for k, v in stats.items()]
    return "\n".join(rows)


def write_readme(tracking: dict, mr: dict, net_perf: dict,
                 gross_perf: dict, start: str, end: str) -> None:
    text = f"""# Tokenised Gold: Tracking Accuracy and Mean-Reversion

Do gold-backed crypto tokens actually track gold, and can you profit when two
of them drift apart?

PAXG (Paxos Gold) and XAUT (Tether Gold) are each backed by one troy ounce of
physical gold, so in theory both should trade at the gold price. This project
measures how closely they do, how quickly gaps between them close, and whether
a mean-reversion strategy on that gap survives transaction costs.

**Data:** daily closes from Yahoo Finance, {start} to {end}
(`PAXG-USD`, `XAUT-USD`, gold futures `GC=F`).

## 1. Tracking accuracy vs gold

{md_table(tracking["PAXG"]).replace("| Metric | Value |", "| PAXG | |")}

{md_table(tracking["XAUT"]).replace("| Metric | Value |", "| XAUT | |")}

![Premium vs gold](output/1_premium_vs_gold.png)

## 2. PAXG vs XAUT: mean reversion

Spread = ln(PAXG) − ln(XAUT). An ADF test checks for stationarity, and an
AR(1) regression (the discrete form of an Ornstein-Uhlenbeck process) gives the
speed of mean reversion and the half-life of a gap.

{md_table(mr)}

## 3. Mean-reversion strategy

Rules: enter when the 30-day z-score of the spread passes ±{ENTRY_Z:g}, exit
when it returns to {EXIT_Z:g}. Signals are lagged one day (no look-ahead).
Fees: {FEE_PER_LEG * 100:.2f}% per token per trade.

**After fees**

{md_table(net_perf)}

**Before fees**

{md_table(gross_perf)}

![Spread and trades](output/2_spread_and_trades.png)

![Equity curve](output/3_equity_curve.png)

## Limitations

- Shorting a token requires borrowing it, which has a cost and is not always
  available. Borrow costs are not modelled.
- Daily closing prices miss intraday dislocations, and Yahoo prices are
  aggregated across exchanges, so they may not be achievable fills.
- Gold futures (`GC=F`) are not spot gold: they include carry and roll between
  contracts, which affects the premium level and adds noise to tracking error.
- Crypto closes at 00:00 UTC while gold futures settle in the US afternoon, so
  token-vs-gold daily returns are not perfectly synchronised. The PAXG vs XAUT
  comparison avoids this.

## How to run

```
pip install yfinance pandas numpy statsmodels matplotlib
python gold_token_analysis.py
```
"""
    Path("README.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------------------------
def main() -> None:
    print("Downloading prices...")
    prices = load_prices()

    # Step 2: tracking vs gold
    tracking, premiums = {}, {}
    for name in ("PAXG", "XAUT"):
        tracking[name], premiums[name] = tracking_vs_gold(
            prices[name], prices["GOLD"], name)
        print_table(f"{name} vs gold", tracking[name])

    # Steps 3 and 4: spread and mean reversion
    spread = build_spread(prices["PAXG"], prices["XAUT"])
    mr = mean_reversion_stats(spread)
    print_table("PAXG vs XAUT spread: mean reversion", mr)

    # Step 5: strategy
    bt = run_strategy(spread)
    net_perf = performance(bt["net"], bt["equity"], bt["position"])
    gross_perf = performance(bt["gross"], bt["equity_gross"], bt["position"])
    print_table("Strategy after fees", net_perf)
    print_table("Strategy before fees", gross_perf)

    # Step 6: charts and README
    make_charts(premiums, bt)
    start, end = str(spread.index[0].date()), str(spread.index[-1].date())
    write_readme(tracking, mr, net_perf, gross_perf, start, end)
    print(f"\nCharts saved to ./{OUTPUT_DIR}/ and README.md written.")


if __name__ == "__main__":
    main()
