"""Run the full optimizer from a terminal: python main.py --help."""

import argparse

from analysis import analyze_portfolio
from holdings import read_holdings
from reporting import export_results


def main():
    parser = argparse.ArgumentParser(description="Import holdings and compare Monte Carlo portfolio allocations.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--holdings", help="Path to a broker holdings export (.csv or .xlsx).")
    source.add_argument("--demo", action="store_true", help="Use explicit synthetic offline data (also the default without a file).")
    parser.add_argument("--decimal", choices=[".", ","], default=".", help="Decimal separator in text cells; no thousands separators.")
    parser.add_argument("--period", choices=["6mo", "1y", "2y", "5y"], default="2y")
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual arithmetic rate as a decimal, e.g. 0.03 for 3%%.")
    parser.add_argument("--output", default="output", help="Report folder (existing report filenames are replaced).")
    args = parser.parse_args()

    try:
        if args.holdings:
            print("1/4 Extracting holdings from the file...")
            holdings = read_holdings(args.holdings, decimal=args.decimal)
            print(holdings.to_string(index=False))
            print("2/4 Downloading completed daily sessions from Yahoo Finance...")
        else:
            holdings = None
            print("1/4 OFFLINE DEMO: synthetic holdings and prices; no API request.")
            print("2/4 Creating reproducible synthetic prices...")
        result = analyze_portfolio(holdings, args.period, args.simulations, args.seed, args.risk_free_rate)
        print("3/4 Estimated annual means and covariance; evaluated portfolio candidates.")
        print(result["comparison"].to_string(float_format=lambda value: f"{value:.4f}"))
        print("\nAllocation weights (decimals):")
        print(result["allocations"].to_string(float_format=lambda value: f"{value:.4f}"))
        for message in result["messages"]:
            print(f"Note: {message}")
        destination = export_results(result, args.output)
        print(f"4/4 Saved charts, tables, inputs, and settings to {destination}")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
