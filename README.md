# Portfolio Optimizer

A step-by-step learning project for a Monte Carlo portfolio optimizer based on
Markowitz's mean-variance framework.

**Current stage:** simple returns and historical mean estimation for one asset.
This first module runs independently; portfolio optimization is not implemented yet.

## Run this step

Use Python 3.10 or newer. From the repository directory:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell with `.venv\Scripts\Activate.ps1`,
or on macOS/Linux with `source .venv/bin/activate`. Then run:

```bash
python -m pip install -r requirements.txt
python main.py
python -m unittest -v
```

The example uses hypothetical prices and requires no API keys or market data download.
Tests use Python's built-in `unittest` framework.

Read [Step 1: Returns](docs/01_returns.md) for the mathematics, a walkthrough of
each code block, exercises, and the planned architecture.

The full project README will be written when the optimizer is complete.
