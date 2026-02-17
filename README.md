# GSK Stock Price Prediction - Monte Carlo Simulation

This project uses **Machine Learning** and **Geometric Brownian Motion** to forecast and simulate 10,000 potential price paths for GlaxoSmithKline (**GSK.L**) over an 18-month investment horizon.

## Overview
1. **Data Management:** Extraction and filtering of GSK and FTSE 100 daily financial dataset since the 01/01 FY2022 until present day with calculation of features used in ML model.
2. **Machine Learning Forecast:** A daily resetting Random Forest model to predict next day closing price change movements in GBX, using FTSE 100 as industry benchmark.
3. **Probability Distribution:** 10,000 path Monte Carlo simulation using the past 390 days of financial data from custom dataset for GSK.

## Structure

### 1. "gsk_prep_data.py"
* **Data Source:** Yahoo Finance ("yfinance").
* **Calculations:** Log returns, rolling volatility, and annual drift.
* **Output:** "gsk_daily.csv", clean dataset of GSK.L and FSTE 100 financial data used for modelling.

### 2. "gsk_ml_model.py"
* **Approach:** Reads "gsk_daily.csv" and uses it's dataset in calculations for metrics like lagged returns, moving average/gap, and rolling volatility.
* **Machine Learning Model:** A Random Forest Regressor model (200 trees, depth 5) that uses walk-forward validation to predict price changes, whilst using FTSE 100 daily return as a comparative benchmark to simulate potential non-statistical (external) shocks to GSK.
* **Visualisation:** Then plots the actual daily closing price of GSK against the ML predicted daily closing price to show accuracy of predictions. 

![ML Price Prediciton](./gsk_images/ml_price_pred_graph.png)

### 3. "gsk_monte_carlo.py"
* **Method:** Geometric Brownian Motion: $S_t = S_0 \exp\left( \left( \mu - \frac{\sigma^2}{2} \right) t + \sigma W_t \right)$.
* **Parameters:** 10,000 simulations over a 390-day (18-month) horizon.
* **Result:** A probability distribution of 500 random paths, current price (Feb 2026), and average price path.

![Monte Carlo Simulation](./gsk_images/monte_carlo_sim_GSK.png)

To see developed analysis read each .py to find fully commented code.