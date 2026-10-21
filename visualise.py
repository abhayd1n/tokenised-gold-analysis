"""
Visual dashboard for the tokenised gold project.

Run this AFTER (or instead of) gold_token_analysis.py:
    python visualise.py

It re-downloads the data, re-runs the analysis and saves three dashboard
images to ./output/, then opens them so you can look at them:

    dashboard_1_the_data.png      -> what the prices look like
    dashboard_2_the_gap.png       -> the PAXG vs XAUT gap and why it mean-reverts
    dashboard_3_the_strategy.png  -> how the trading rule makes or loses money
"""

import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Reuse everything from the main analysis so the numbers always match
import gold_token_analysis as g

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

COLOURS = {"PAXG": "#d4a017", "XAUT": "#1f77b4", "GOLD": "#555555"}


def last_n_days(series_or_df, n):
    cutoff = series_or_df.index.max() - pd.Timedelta(days=n)
    return series_or_df[series_or_df.index >= cutoff]


# ---------------------------------------------------------------------------
# DASHBOARD 1: THE DATA
# ---------------------------------------------------------------------------
def dashboard_data(prices, premiums):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("1. The data: do gold tokens follow gold?", fontsize=14,
                 fontweight="bold")

    # (a) Full price history of all three
    ax = axes[0, 0]
    for name in ("GOLD", "PAXG", "XAUT"):
        ax.plot(prices[name].index, prices[name], label=name,
                color=COLOURS[name], linewidth=0.9)
    ax.set_title("(a) Price of each, since 2020 (US$ per ounce)")
    ax.legend()

    # (b) Zoom: last 90 days, so you can see the small differences
    ax = axes[0, 1]
    for name in ("GOLD", "PAXG", "XAUT"):
        s = last_n_days(prices[name], 90)
        ax.plot(s.index, s, label=name, color=COLOURS[name], linewidth=1.2,
                marker="." if name == "GOLD" else None)
    ax.set_title("(b) Zoom: last 90 days (gold has gaps at weekends)")
    ax.legend()

    # (c) Premium of each token vs gold
    ax = axes[1, 0]
    for name, prem in premiums.items():
        ax.plot(prem.index, prem * 100, label=name, color=COLOURS[name],
                linewidth=0.7, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.7)
    ax.axhline(-0.5, color="red", linestyle="--", linewidth=0.7,
               label="±0.5% band")
    ax.set_title("(c) Token price vs gold futures: % above (+) or below (−)")
    ax.set_ylabel("%")
    ax.legend()

    # (d) Daily return scatter: token vs gold, with the regression line
    ax = axes[1, 1]
    for name in ("PAXG", "XAUT"):
        both = pd.concat([prices[name].rename(name),
                          prices["GOLD"].rename("GOLD")],
                         axis=1, join="inner").dropna()
        r = np.log(both).diff().dropna() * 100
        ax.scatter(r["GOLD"], r[name], s=4, alpha=0.35, color=COLOURS[name],
                   label=f"{name} daily returns")
        fit = sm.OLS(r[name], sm.add_constant(r["GOLD"])).fit()
        x = np.linspace(r["GOLD"].min(), r["GOLD"].max(), 50)
        ax.plot(x, fit.params["const"] + fit.params["GOLD"] * x,
                color=COLOURS[name], linewidth=1.5,
                label=f"{name} fit: beta {fit.params['GOLD']:.2f}, "
                      f"R² {fit.rsquared:.2f}")
    lim = 4
    ax.plot([-lim, lim], [-lim, lim], color="black", linestyle=":",
            linewidth=1, label="perfect tracking (45° line)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Gold daily return (%)")
    ax.set_ylabel("Token daily return (%)")
    ax.set_title("(d) Same-day moves: token vs gold")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DASHBOARD 2: THE GAP BETWEEN PAXG AND XAUT
# ---------------------------------------------------------------------------
def dashboard_gap(spread, bt, mr):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("2. The gap: PAXG vs XAUT, and why it snaps back",
                 fontsize=14, fontweight="bold")
    bps = spread * 10_000

    # (a) Spread with its rolling average and ±2 standard-deviation bands
    ax = axes[0, 0]
    mean = bps.rolling(g.Z_WINDOW).mean()
    std = bps.rolling(g.Z_WINDOW).std()
    ax.plot(bps.index, bps, color="grey", linewidth=0.6, label="gap (bps)")
    ax.plot(mean.index, mean, color="black", linewidth=1,
            label=f"{g.Z_WINDOW}-day average")
    ax.fill_between(mean.index, mean - g.ENTRY_Z * std,
                    mean + g.ENTRY_Z * std, color="orange", alpha=0.2,
                    label=f"±{g.ENTRY_Z:g} st. dev. band")
    ax.set_title("(a) Gap between PAXG and XAUT (100 bps = 1%)")
    ax.set_ylabel("bps")
    ax.legend(fontsize=7)

    # (b) Distribution of the gap
    ax = axes[0, 1]
    ax.hist(bps.dropna(), bins=80, color="grey", alpha=0.7, density=True)
    x = np.linspace(bps.min(), bps.max(), 200)
    mu, sd = bps.mean(), bps.std()
    ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color="red", linewidth=1.2, label="normal curve (same mean/st. dev.)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"(b) How big the gap usually is "
                 f"(mean {mu:.1f} bps, st. dev. {sd:.1f} bps)")
    ax.set_xlabel("gap (bps)")
    ax.legend(fontsize=7)

    # (c) The mean-reversion regression: yesterday's gap vs today's change
    ax = axes[1, 0]
    d = pd.concat([bps.shift(1), bps.diff()], axis=1).dropna()
    d.columns = ["yesterday", "change"]
    ax.scatter(d["yesterday"], d["change"], s=4, alpha=0.3, color="purple")
    fit = sm.OLS(d["change"], sm.add_constant(d["yesterday"])).fit()
    xs = np.linspace(d["yesterday"].min(), d["yesterday"].max(), 50)
    ax.plot(xs, fit.params["const"] + fit.params["yesterday"] * xs,
            color="black", linewidth=1.5,
            label=f"slope b = {fit.params['yesterday']:.3f}  ->  "
                  f"half-life {mr['half-life (days)']:.1f} days")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("yesterday's gap (bps)")
    ax.set_ylabel("today's change in gap (bps)")
    ax.set_title("(c) Big gap yesterday -> it shrinks today (negative slope)")
    ax.legend(fontsize=8)

    # (d) What a 2.4-day half-life looks like: a 100 bps gap decaying
    ax = axes[1, 1]
    theta = mr["OU speed theta (per day)"]
    days = np.arange(0, 15)
    if not np.isnan(theta):
        ax.plot(days, 100 * np.exp(-theta * days), marker="o", color="green",
                label="expected gap if it starts at 100 bps")
        hl = mr["half-life (days)"]
        ax.axhline(50, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(hl, color="red", linestyle="--", linewidth=0.8,
                   label=f"half gone after {hl:.1f} days")
    ax.set_xlabel("days after the gap opens")
    ax.set_ylabel("expected gap (bps)")
    ax.set_title("(d) The Ornstein-Uhlenbeck model: how fast a gap closes")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DASHBOARD 3: THE STRATEGY
# ---------------------------------------------------------------------------
def dashboard_strategy(bt):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("3. The strategy: betting the gap closes", fontsize=14,
                 fontweight="bold")

    # (a) z-score with entry thresholds, shading the days we're in a trade
    ax = axes[0, 0]
    ax.plot(bt.index, bt["z"], color="grey", linewidth=0.5, label="z-score")
    ax.axhline(g.ENTRY_Z, color="red", linestyle="--", linewidth=0.8,
               label=f"enter at ±{g.ENTRY_Z:g}")
    ax.axhline(-g.ENTRY_Z, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    in_long = bt["position"] == 1
    in_short = bt["position"] == -1
    ax.fill_between(bt.index, -5, 5, where=in_long, color="green", alpha=0.15,
                    label="in a long-gap trade")
    ax.fill_between(bt.index, -5, 5, where=in_short, color="red", alpha=0.15,
                    label="in a short-gap trade")
    ax.set_ylim(-5, 5)
    ax.set_title("(a) How unusual the gap is (z-score) and when we trade")
    ax.legend(fontsize=7, loc="upper left")

    # (b) Zoom on one real trade: the biggest entry in the last two years
    ax = axes[0, 1]
    recent = last_n_days(bt, 730)
    entries = recent[(recent["signal"] != 0)
                     & (recent["signal"].shift(1).fillna(0) == 0)]
    if len(entries):
        entry_day = entries["z"].abs().idxmax()
        window = bt.loc[entry_day - pd.Timedelta(days=15):
                        entry_day + pd.Timedelta(days=15)]
        ax.plot(window.index, window["spread"] * 10_000, color="black",
                marker=".", label="gap (bps)")
        ax.axvline(entry_day, color="orange", linewidth=1.5,
                   label="signal day (trade opens next day)")
        ax2 = ax.twinx()
        ax2.bar(window.index, window["position"], color="green", alpha=0.25,
                width=0.8, label="position held")
        ax2.set_ylim(-3, 3)
        ax2.set_ylabel("position (+1 long gap, −1 short gap)")
        ax2.grid(False)
        ax.set_title(f"(b) One trade up close: signal on {entry_day.date()}")
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        ax.set_title("(b) No trades in the last two years")

    # (c) Equity curve before and after fees
    ax = axes[1, 0]
    ax.plot(bt.index, bt["equity_gross"], label="before fees", linewidth=1)
    ax.plot(bt.index, bt["equity"], label="after fees", linewidth=1.5)
    ax.axhline(1, color="black", linewidth=0.6)
    ax.set_title("(c) Growth of £1 put into each leg")
    ax.legend()

    # (d) Drawdown: how far below its previous best the strategy fell
    ax = axes[1, 1]
    dd = (bt["equity"] / bt["equity"].cummax() - 1) * 100
    ax.fill_between(dd.index, dd, 0, color="red", alpha=0.4)
    ax.set_title(f"(d) Drawdown after fees (worst: {dd.min():.1f}%)")
    ax.set_ylabel("% below previous peak")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def main():
    print("Downloading prices and rebuilding the analysis...")
    prices = g.load_prices()

    premiums = {}
    for name in ("PAXG", "XAUT"):
        _, premiums[name] = g.tracking_vs_gold(prices[name], prices["GOLD"], name)

    spread = g.build_spread(prices["PAXG"], prices["XAUT"])
    mr = g.mean_reversion_stats(spread)
    bt = g.run_strategy(spread)

    g.OUTPUT_DIR.mkdir(exist_ok=True)
    files = []
    for fname, fig in [
        ("dashboard_1_the_data.png", dashboard_data(prices, premiums)),
        ("dashboard_2_the_gap.png", dashboard_gap(spread, bt, mr)),
        ("dashboard_3_the_strategy.png", dashboard_strategy(bt)),
    ]:
        path = g.OUTPUT_DIR / fname
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files.append(path)
        print(f"  saved {path}")

    # Open the images automatically on Windows
    if os.name == "nt":
        for path in files:
            os.startfile(path.resolve())


if __name__ == "__main__":
    main()
