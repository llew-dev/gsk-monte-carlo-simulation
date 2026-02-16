import numpy as np
import yfinance as yf

# fetching GSK DAILY financial data since 1/1/2022 from Yahoo Finance through yfinance
ftse = yf.download("^FTSE", start="2022-01-01", progress=False)
# GSK DAILY Close price
ftse = ftse[["Close"]]
# daily log return from previous day
ftse["ftse_return"] = np.log(ftse["Close"]/ftse["Close"].shift(1))
# assigning daily log return to variable ftse
ftse = ftse[["ftse_return"]]

# repeating the same process for GSK
ticker = ("GSK.L")
gsk_daily = yf.download(ticker, start="2022-01-01", progress=False)
gsk_daily = gsk_daily[["Close"]]
gsk_daily["log_return"]=np.log(gsk_daily["Close"]/gsk_daily["Close"].shift(1))

# merges the two datasets ftse and gsk_daily
gsk_daily = gsk_daily.merge(ftse, left_index=True, right_index=True, how="left")

# removed the blank data from the very beginning of the dataset
gsk_daily = gsk_daily.dropna()
print(gsk_daily.head())
print("\nSummary Statistics:")
print(gsk_daily["log_return"].describe())

# calculating Drift or Avg Weekly Returns
daily_drift = gsk_daily["log_return"].mean()
print(f"\nWeekly Drift:", round(daily_drift, 6))

# converting to yearly to work better with long term investment horizon
days_per_year = 252
annual_drift = daily_drift * days_per_year
print(f"\nAnnual Drift:", round(annual_drift, 6))

# 52-week rolling volatility or Std Dev over 52-week period
gsk_daily["rolling_vol"] = gsk_daily["log_return"].rolling(window=252).std()
print(gsk_daily[["log_return", "rolling_vol"]].tail())

# annualising volatility
gsk_daily["annual_vol"] = gsk_daily["rolling_vol"] * np.sqrt(252)
print(gsk_daily[["rolling_vol", "annual_vol"]].tail)

gsk_daily["target"] = gsk_daily["log_return"].shift(-1)

gsk_daily = gsk_daily.dropna()

gsk_daily.to_csv("gsk_daily.csv")
print("Saved dataset to gsk_daily.csv")

sigma_weekly = gsk_daily["log_return"].std()