import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail FMCG Inventory & Forecast Dashboard", layout="wide")
st.title("📦 Retail FMCG — Inventory & Sales Forecast Dashboard")

@st.cache_data
def load_data():
    hist = pd.read_csv("historical_daily_sales.csv", parse_dates=["order_day"])
    future = pd.read_csv("future_forecasts_30days.csv", parse_dates=["ds"])
    inv = pd.read_csv("inventory_analysis_summary.csv")
    results = pd.read_csv("forecast_results_all_products.csv")
    return hist, future, inv, results

hist, future, inv, results = load_data()

products = sorted(hist["product_name"].unique())
selected_product = st.sidebar.selectbox("Select Product", products)

st.sidebar.markdown("---")
st.sidebar.subheader("Inventory Snapshot")
row = inv[inv["product_name"] == selected_product].iloc[0]
st.sidebar.metric("Stock Turnover Ratio", f"{row['stock_turnover_ratio']:.1f}")
st.sidebar.metric("Days of Inventory Outstanding", f"{row['dio']:.1f} days")
st.sidebar.metric("Sell-Through Rate", f"{row['sell_through_rate']*100:.1f}%")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{selected_product}: Historical Sales + 30-Day Forecast")
    h = hist[hist["product_name"] == selected_product]
    f = future[future["product_name"] == selected_product]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(h["order_day"].tail(120), h["daily_qty"].tail(120), label="Historical (last 120 days)")
    ax.plot(f["ds"], f["yhat"], label="Forecast", color="orange", linestyle="--")
    ax.fill_between(f["ds"], f["yhat_lower"], f["yhat_upper"], color="orange", alpha=0.2, label="Confidence interval")
    ax.legend()
    ax.set_ylabel("Units sold per day")
    st.pyplot(fig)

with col2:
    st.subheader("Model Accuracy")
    prod_results = results[results["Product"] == selected_product] if "Product" in results.columns else results
    st.dataframe(prod_results, hide_index=True)

st.markdown("---")
st.subheader("Full Inventory Summary — All Products")
st.dataframe(inv, hide_index=True)

st.markdown("---")
st.caption("Retail FMCG Inventory Optimization Project — Data Analyst + Data Scientist pipeline")
