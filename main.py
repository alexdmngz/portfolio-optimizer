"""Run the first learning step with a small, reproducible price series."""

from returns import calculate_returns, estimate_mean_return


def main():
    # Hypothetical prices at three consecutive monthly observations.
    prices = [100, 110, 99]

    returns = calculate_returns(prices)
    mean_return = estimate_mean_return(returns)
    cumulative_return = prices[-1] / prices[0] - 1

    print("Portfolio Optimizer - Step 1: Returns")
    print("Example prices:", prices)
    print("Monthly returns:", returns)
    print(f"Estimated mean monthly return: {mean_return:.2%}")
    print(f"Cumulative return over two months: {cumulative_return:.2%}")


if __name__ == "__main__":
    main()
