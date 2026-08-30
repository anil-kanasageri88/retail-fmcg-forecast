# 📦 Retail FMCG — Inventory Optimization & Sales Forecasting

An end-to-end data analytics and machine learning project that diagnoses inventory inefficiencies in a retail FMCG business and forecasts future product demand to support proactive inventory planning.

**🔗 Live App:** [retail-fmcg-forecast.streamlit.app](https://retail-fmcg-forecast-rattehtv6fvwjqoz7afa29.streamlit.app)

---

## Business Problem

A retail FMCG company sells 5 core products (Soap, Shampoo, Oil, Biscuits, Rice) across multiple regions. Inaccurate demand estimation leads to two costly problems: **overstocking** (tied-up capital, storage costs, waste) and **understocking** (stockouts, lost sales, dissatisfied customers). This project analyzes historical sales/inventory data to diagnose these issues and builds forecasting models to support better inventory decisions.

## Project Structure

The project follows a two-phase workflow mirroring real-world retail analytics roles:

### Phase 1 — Data Analyst
- **Data Cleaning:** missing values, duplicates, outlier detection (IQR method), data type conversion
- **Exploratory Data Analysis:** product/category/region-wise sales, monthly trends, seasonality, discount impact, revenue analysis
- **Inventory Analysis:** stock turnover ratio, Days of Inventory Outstanding (DIO), sell-through rate, fast/slow-moving and dead stock classification
- **Statistical Analysis:** probability & conditional probability of stockouts, binomial modeling, confidence intervals, hypothesis testing, A/B testing across regions

### Phase 2 — Data Scientist (Feature-Based Models)
- **Feature Engineering:** lag features, rolling averages, seasonal features, stock coverage days, sell-through rate (leakage-free)
- **Models compared:** Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Best model:** CatBoost (R² = 0.115, RMSE = 9.52, MAPE = 48.4%)

### Phase 3 — Time Series Forecasting
- **Models compared:** ARIMA, SARIMA, Prophet, LSTM (deep learning)
- **Final model:** Prophet — chosen for comparable accuracy to LSTM, much faster training, and more interpretable trend/seasonality components
- 30-day forward forecast generated per product

### Deployment
- Interactive **Streamlit** dashboard: select a product, view historical sales + 30-day forecast with confidence intervals, inventory health metrics, and model accuracy comparison

## Key Findings

- All 5 products turn over inventory every ~10 days on average — healthy, fast-moving FMCG behavior with no major overstock/understock crisis at the product level
- **Discounts have no statistically significant effect** on quantity sold (hypothesis test, p = 0.76)
- **No significant regional difference** in sales performance (A/B test, East vs. West, p = 0.95)
- Stockouts are rare (0.16% of orders) and evenly distributed across categories
- A data quality issue was identified and corrected: the raw `product_id` field was inconsistently mapped to products and could not be used as a reliable grouping key — analysis was redone using `product_name` instead
- A data leakage bug was caught and fixed in feature engineering (`sell_through_rate` initially used same-day demand; corrected to use only prior-day data)
- LSTM was evaluated but did not meaningfully outperform Prophet, given the dataset's size and weak day-to-day predictability — Prophet was retained as the simpler, equally effective choice

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `statsmodels` (ARIMA/SARIMA) · `Prophet` · `TensorFlow/Keras` (LSTM) · `Streamlit` · `Matplotlib` / `Seaborn`

## Repository Contents

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard application |
| `requirements.txt` | Python dependencies |
| `prophet_models.pkl` | Trained Prophet forecasting models (one per product) |
| `historical_daily_sales.csv` | Cleaned daily sales panel used for charts |
| `future_forecasts_30days.csv` | 30-day forward forecast per product |
| `inventory_analysis_summary.csv` | Inventory health metrics per product |
| `forecast_results_all_products.csv` | Model accuracy comparison (ARIMA/SARIMA/Prophet) |

## Running Locally

```bash
git clone https://github.com/anil-kanasageri88/retail-fmcg-forecast.git
cd retail-fmcg-forecast
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Anil Kanasageri** — Data Science / Analytics
[GitHub](https://github.com/anil-kanasageri88)
