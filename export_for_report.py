"""
Export everything needed for the written report.

Run from the same folder as gold_token_analysis.py:
    python export_for_report.py

It re-runs the full analysis, adds a split-period ("alpha decay") analysis,
and saves:
    report_data/prices.csv       daily prices for PAXG, XAUT and gold futures
    report_data/backtest.csv     spread, z-score, positions, returns, equity
    report_data/results.json     every summary statistic used in the report
    report_data.zip              all of the above in one file to upload
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import gold_token_analysis as g

OUT = Path("report_data")

# Sub-periods for the alpha-decay analysis
PERIODS = {
    "2020-2022": ("2020-01-01", "2022-12-31"),
    "2023-2024": ("2023-01-01", "2024-12-31"),
    "2025-present": ("2025-01-01", "2100-01-01"),
}


def to_plain(obj):
    """Convert numpy numbers to plain Python so they can be saved as JSON."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    return obj


def slice_stats(bt: pd.DataFrame, spread: pd.Series, start: str, end: str) -> dict:
    """Strategy and mean-reversion statistics for one sub-period."""
    b = bt.loc[start:end]
    s = spread.loc[start:end]
    if len(b) < 60:
        return {}
    net_eq = np.exp(b["net"].cumsum())
    gross_eq = np.exp(b["gross"].cumsum())
    mr = g.mean_reversion_stats(s)
    return {
        "start": str(b.index[0].date()),
        "end": str(b.index[-1].date()),
        "days": len(b),
        "spread st. dev.": s.std(),
        "mean absolute spread": s.abs().mean(),
        "half-life (days)": mr["half-life (days)"],
        "ADF p-value": mr["ADF p-value"],
        "net": g.performance(b["net"], net_eq, b["position"]),
        "gross": g.performance(b["gross"], gross_eq, b["position"]),
        "fees paid (return points)": b["costs"].sum(),
    }


def main():
    print("Re-running the analysis...")
    prices = g.load_prices()

    tracking, premiums = {}, {}
    for name in ("PAXG", "XAUT"):
        tracking[name], premiums[name] = g.tracking_vs_gold(
            prices[name], prices["GOLD"], name)

    spread = g.build_spread(prices["PAXG"], prices["XAUT"])
    mr = g.mean_reversion_stats(spread)
    bt = g.run_strategy(spread)
    net_perf = g.performance(bt["net"], bt["equity"], bt["position"])
    gross_perf = g.performance(bt["gross"], bt["equity_gross"], bt["position"])

    # Alpha decay: the same statistics for each sub-period and each year
    periods = {name: slice_stats(bt, spread, a, b)
               for name, (a, b) in PERIODS.items()}
    years = {}
    for yr in sorted(set(bt.index.year)):
        st = slice_stats(bt, spread, f"{yr}-01-01", f"{yr}-12-31")
        if st:
            years[str(yr)] = st

    # Print the alpha-decay table so you can see it straight away
    print("\nAlpha decay by period (after fees)")
    print("----------------------------------")
    for name, st in periods.items():
        if st:
            n = st["net"]
            print(f"  {name:<13} Sharpe {n['Sharpe ratio']:>6.2f}   "
                  f"annual return {n['annual return'] * 100:>6.2f}%   "
                  f"trades {n['number of trades']:>3}   "
                  f"spread st. dev. {st['spread st. dev.'] * 10_000:>6.1f} bps   "
                  f"half-life {st['half-life (days)']:.2f} days")

    # Save the data
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(prices).to_csv(OUT / "prices.csv", index_label="date")
    pd.DataFrame(premiums).add_prefix("premium_").to_csv(
        OUT / "premiums.csv", index_label="date")
    bt.to_csv(OUT / "backtest.csv", index_label="date")

    results = {
        "settings": {
            "start date": g.START_DATE,
            "tickers": g.TICKERS,
            "z window": g.Z_WINDOW,
            "entry z": g.ENTRY_Z,
            "exit z": g.EXIT_Z,
            "fee per leg": g.FEE_PER_LEG,
        },
        "data coverage": {
            name: {"days": len(s), "first": str(s.index[0].date()),
                   "last": str(s.index[-1].date())}
            for name, s in prices.items()
        },
        "tracking vs gold": tracking,
        "mean reversion": mr,
        "strategy net": net_perf,
        "strategy gross": gross_perf,
        "periods": periods,
        "years": years,
    }
    (OUT / "results.json").write_text(json.dumps(to_plain(results), indent=2),
                                      encoding="utf-8")

    zip_path = shutil.make_archive("report_data", "zip", OUT)
    print(f"\nSaved everything to {OUT}/ and zipped it to {Path(zip_path).name}")
    print("Upload report_data.zip to Claude.")


if __name__ == "__main__":
    main()
