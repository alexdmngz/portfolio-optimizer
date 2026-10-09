"""A small local interface. Run with: python -m streamlit run app.py."""

from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st

from analysis import analyze_portfolio
from holdings import read_holdings
from reporting import plot_portfolios


st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer")
st.write("Import your holdings, explore risk and return, and compare portfolio allocations.")
st.caption("Long-only stocks and ETFs · One quote currency · Historical estimates · No order execution")

with st.sidebar:
    st.header("Inputs")
    mode = st.radio("Data source", ["Offline demo (synthetic)", "Import broker file + Yahoo prices"])
    period = st.selectbox("Price history", ["6mo", "1y", "2y", "5y"], index=2)
    count = st.number_input("Random portfolios", min_value=1, max_value=200000, value=10000, step=1000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    risk_free_percent = st.number_input("Annual risk-free rate (%)", min_value=-10.0, max_value=100.0, value=0.0, step=0.25)
    st.caption("Set this rate explicitly in the same currency as your assets. 0% is a demo default.")

holdings = None
ready = True
if mode == "Import broker file + Yahoo prices":
    st.write("Export your current positions from your broker, then upload the CSV or XLSX file.")
    st.caption("Headers on row 1: Ticker/Symbol/Símbolo and Quantity/Shares/Cantidad, or Weight/Peso. "
               "Quantities take priority if both are present. Use Yahoo symbols; ISINs and PDFs need conversion first.")
    uploaded = st.file_uploader("Holdings export", type=["csv", "xlsx"])
    decimal = st.selectbox("Decimal separator for text numbers", [".", ","])
    ready = uploaded is not None
    if uploaded is not None:
        try:
            uploaded.seek(0)
            holdings = read_holdings(uploaded, uploaded.name, decimal)
            st.subheader("Extracted holdings — check before running")
            st.dataframe(holdings, hide_index=True, width="stretch")
        except (ValueError, OSError, ImportError) as error:
            ready = False
            st.error(str(error))
else:
    st.info("The demo uses synthetic holdings and prices and works offline.")

if st.button("Run optimization", type="primary", disabled=not ready):
    st.session_state.pop("result", None)
    try:
        with st.spinner("Loading prices and evaluating portfolio weights..."):
            st.session_state["result"] = analyze_portfolio(holdings, period, int(count), int(seed), risk_free_percent / 100)
    except (ValueError, OSError) as error:
        st.error(str(error))

if "result" in st.session_state:
    result = st.session_state["result"]
    settings = result["settings"]
    st.subheader("Last completed run")
    st.caption(f"{settings['source']} · {settings['currency']} · Valuation: {settings['valuation_date']} · "
               f"{settings['observations']} daily returns · Seed: {settings['seed']} · "
               f"Risk-free rate: {settings['annual_risk_free_rate']:.2%}. Run again after changing inputs.")
    for message in result["messages"]:
        st.info(message)
    st.dataframe(result["comparison"].style.format({"expected_return": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.3f}"}, na_rep="Undefined"), width="stretch")
    figure = plot_portfolios(result)
    st.pyplot(figure)
    picture = BytesIO()
    figure.savefig(picture, format="png", dpi=160)
    plt.close(figure)
    st.subheader("Allocation weights")
    st.dataframe(result["allocations"].style.format("{:.2%}"), width="stretch")
    left, middle, right = st.columns(3)
    left.download_button("Download allocations", result["allocations"].to_csv().encode("utf-8"), "allocations.csv", "text/csv")
    middle.download_button("Download comparison", result["comparison"].to_csv().encode("utf-8"), "comparison.csv", "text/csv")
    right.download_button("Download chart", picture.getvalue(), "portfolios.png", "image/png")
    with st.expander("How the calculation works"):
        st.write("1. Quantities × closing prices give current market values; divide by their total to get weights.")
        st.write("2. Adjusted price ratios give daily returns. Their means and sample covariance describe the assets.")
        st.write("3. Annualized return = wᵀμ; volatility = √(wᵀΣw); Sharpe = (return − risk-free rate) / volatility.")
        st.write("4. Dirichlet sampling generates non-negative weights that sum to one. We also test the current, equal-weight, and single-asset portfolios.")
        st.write("5. Best sampled Sharpe and lowest sampled volatility are historical comparisons, not guaranteed optima or future results.")
    with st.expander("Run settings"):
        st.json(settings)
