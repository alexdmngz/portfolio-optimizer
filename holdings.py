"""Extract a long-only portfolio from a broker's CSV or Excel export."""

from pathlib import Path
import re
import unicodedata
from zipfile import BadZipFile

import numpy as np
import pandas as pd


COLUMN_NAMES = {
    "ticker": {"ticker", "symbol", "simbolo", "simbolo bursatil"},
    "quantity": {"quantity", "qty", "shares", "units", "cantidad", "unidades", "titulos"},
    "weight": {"weight", "weight %", "allocation", "allocation %", "peso", "peso %", "porcentaje"},
}


def normalize_header(value):
    """Ignore capitalization, accents, underscores, and repeated spaces."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(letter for letter in value if not unicodedata.combining(letter))
    return " ".join(value.lower().replace("_", " ").strip().split())


def parse_numbers(values, decimal=".", allow_percent=False):
    """Parse one explicit decimal convention; reject ambiguous thousands separators."""
    if decimal not in (".", ","):
        raise ValueError("The decimal separator must be '.' or ','.")
    text = values.astype(str).str.strip()
    percentages = text.str.endswith("%")
    if percentages.any():
        if not allow_percent or not percentages.all():
            raise ValueError("Use percent signs consistently, and only for weights.")
        text = text.str.removesuffix("%").str.strip()

    # Excel numeric cells are already numbers, independent of their display format.
    numbers = []
    for original, item in zip(values, text):
        if isinstance(original, (int, float, np.number)) and not isinstance(original, bool):
            number = float(original)
        else:
            pattern = rf"[+-]?(?:\d+(?:{re.escape(decimal)}\d+)?|{re.escape(decimal)}\d+)"
            if not re.fullmatch(pattern, item):
                raise ValueError(f"Invalid number '{item}'. Use decimal '{decimal}' without thousands separators.")
            number = float(item.replace(decimal, "."))
        numbers.append(number)
    result = np.asarray(numbers, dtype=float)
    if not np.isfinite(result).all() or (result < 0).any() or not np.isfinite(result.sum()) or result.sum() <= 0:
        raise ValueError("Holdings must be finite, non-negative, and have a positive total.")
    if percentages.any():
        result = result / 100
    return result


def read_holdings(source, filename=None, decimal="."):
    """Read a path or uploaded file. Prefer quantities when both quantities and weights exist."""
    name = filename or str(getattr(source, "name", source))
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(source, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        try:
            table = pd.read_excel(source, engine="openpyxl")
        except (BadZipFile, KeyError) as error:
            raise ValueError("Cannot read this XLSX file. Export the holdings table again from your broker.") from error
    else:
        raise ValueError("Upload a CSV or XLSX holdings table, with headers on the first row.")
    return clean_holdings(table, decimal)


def clean_holdings(table, decimal="."):
    """Map known headers, validate values, and combine repeated positions."""
    table = table.dropna(how="all").copy()
    renamed = {}
    for column in table.columns:
        header = normalize_header(column)
        # Pandas appends .1, .2, ... when an export repeats a header.
        header = re.sub(r"\.\d+$", "", header)
        for standard, aliases in COLUMN_NAMES.items():
            if header in aliases:
                if standard in renamed.values():
                    raise ValueError(f"Several columns match '{standard}'; keep one unambiguous column.")
                renamed[column] = standard
    table = table.rename(columns=renamed)
    if "ticker" not in table:
        raise ValueError("No ticker column found. Use Ticker, Symbol, or Símbolo with Yahoo symbols.")
    if "quantity" in table:
        position_column = "quantity"
    elif "weight" in table:
        position_column = "weight"
    else:
        raise ValueError("No holdings found. Include Quantity/Cantidad or Weight/Peso.")
    if table.empty or table[["ticker", position_column]].isna().any().any():
        raise ValueError("Every holding needs a ticker and a quantity or weight.")

    tickers = table["ticker"].astype(str).str.strip().str.upper()
    for ticker in tickers:
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.^=\-]{0,29}", ticker):
            raise ValueError(f"Invalid ticker '{ticker}'. Use the Yahoo Finance symbol, not a company name.")
        if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", ticker):
            raise ValueError(f"'{ticker}' looks like an ISIN. Replace it with its exchange-specific Yahoo symbol.")

    values = parse_numbers(table[position_column], decimal, allow_percent=position_column == "weight")
    if position_column == "weight":
        has_percent_signs = table[position_column].astype(str).str.strip().str.endswith("%").any()
        if not has_percent_signs and np.isclose(values.sum(), 100, atol=0.1, rtol=0):
            values = values / 100
        if not np.isclose(values.sum(), 1, atol=0.001, rtol=0):
            raise ValueError("Weights must sum to 1 (fractions) or 100 (percentages).")
        values = values / values.sum()

    holdings = pd.DataFrame({"ticker": tickers.to_numpy(), position_column: values})
    holdings = holdings.groupby("ticker", sort=False, as_index=False)[position_column].sum()
    if len(holdings) > 50:
        raise ValueError("This learning application supports at most 50 assets per portfolio.")
    return holdings


def current_weights(holdings, last_close):
    """Value quantities at unadjusted closing prices; preserve the file's ticker order."""
    if "weight" in holdings:
        return holdings["weight"].to_numpy(dtype=float)
    prices = last_close.reindex(holdings["ticker"]).to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("A valid closing price is required for every holding.")
    values = holdings["quantity"].to_numpy(dtype=float) * prices
    if not np.isfinite(values).all() or not np.isfinite(values.sum()) or values.sum() <= 0:
        raise ValueError("The portfolio's market value must be positive and finite.")
    return values / values.sum()
