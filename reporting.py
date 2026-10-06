"""Display results without recalculating financial metrics."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def plot_portfolios(result):
    """Draw a sampled opportunity set and allocation comparison."""
    metrics = result["metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    finite = np.isfinite(metrics["sharpe"])
    axes[0].scatter(metrics.loc[~finite, "volatility"], metrics.loc[~finite, "expected_return"],
                    s=9, color="lightgray", alpha=0.5)
    if finite.any():
        cloud = axes[0].scatter(metrics.loc[finite, "volatility"], metrics.loc[finite, "expected_return"],
                                c=metrics.loc[finite, "sharpe"], cmap="viridis", s=9, alpha=0.55)
        figure.colorbar(cloud, ax=axes[0], label="Annualized Sharpe ratio")
    markers = ["o", "s", "D", "*"]
    colors = ["#e76f51", "#2563eb", "#7c3aed", "#d19a00"]
    for (label, index), marker, color in zip(result["selected"].items(), markers, colors):
        point = metrics.loc[index]
        axes[0].scatter(point["volatility"], point["expected_return"], marker=marker,
                        color=color, s=150, edgecolors="black", label=label, zorder=3)
    axes[0].set(xlabel="Annualized volatility", ylabel="Estimated annual arithmetic return",
                title="Sampled portfolios — not an exact frontier")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    result["allocations"].T.plot.bar(stacked=True, ax=axes[1], rot=20, colormap="tab20")
    axes[1].set(ylabel="Portfolio weight", title="Allocation comparison", ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=7, ncol=2, title="Ticker")
    if result["settings"]["source"] == "synthetic demo":
        figure.suptitle("OFFLINE DEMO · SYNTHETIC DATA", fontweight="bold")
    return figure


def export_results(result, directory):
    """Save a reproducible local report. User holdings and downloaded data stay out of git."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result["comparison"].to_csv(directory / "comparison.csv", index_label="portfolio")
    result["allocations"].to_csv(directory / "allocations.csv")
    result["prices"].to_csv(directory / "adjusted_prices.csv", index_label="date")
    result["returns"].to_csv(directory / "daily_returns.csv", index_label="date")
    result["holdings"].to_csv(directory / "imported_holdings.csv", index=False)
    result["last_close"].to_csv(directory / "valuation_prices.csv", header=["close"], index_label="ticker")
    candidates = pd.DataFrame(result["weights"], columns=[f"weight_{symbol}" for symbol in result["allocations"].index])
    pd.concat([result["metrics"], candidates], axis=1).to_csv(directory / "candidates.csv", index=False)
    metadata = {**result["settings"], "notes": result["messages"]}
    (directory / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    figure = plot_portfolios(result)
    figure.savefig(directory / "portfolios.png", dpi=160)
    plt.close(figure)
    return directory.resolve()
