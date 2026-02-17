import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# reads the data saved from the prep data file
gsk = pd.read_csv("gsk_daily.csv", index_col=0, parse_dates=True)

# converting .csv files into usable integers fpr calculations
gsk["Close"] = pd.to_numeric(gsk["Close"], errors="coerce")
gsk["log_return"] = pd.to_numeric(gsk["log_return"], errors="coerce")
gsk["ftse_return"] = pd.to_numeric(gsk["ftse_return"], errors="coerce")

# creating lagged returns in order to learn how market returns changed weekly
gsk["ret_1"] = gsk["log_return"].shift(1)
gsk["ret_5"] = gsk["log_return"].shift(5)
gsk["ret_21"] = gsk["log_return"].shift(21)

# rolling volatility in past 5 and 21 days
gsk["vol_5"] = gsk["log_return"].rolling(5).std()
gsk["vol_21"] = gsk["log_return"].rolling(21).std()

# moving avg is the avg prices over this timespan
# moving avg gap is the price differential between current price and the daily moving avg
# pos moving avg gap shows the current price for that day has opened higher than the yesterday's close.
# neg ma indicates a downwards trend, so the avg price is falling
# pos ma indicates upwards trend, so avg price is increasing
gsk["ma_21"] = gsk["Close"].rolling(21).mean()
gsk["ma_gap"] = (gsk["Close"] - gsk["ma_21"]) / gsk["ma_21"]

# remove unusable data, i.e all the NaNs from the beginning of the data set
gsk = gsk.dropna()

# importing main machine learning model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# all the metrics that are going to be used
features = [
    "ret_1", "ret_5", "ret_21",
    "vol_5", "vol_21",
    "ma_gap", "ftse_return"
]

x = gsk[features]
y = gsk["target"]

# empty lists to fill with the ml's predictions, the actual data for comparison and the
# dates so data can be sorted chronologically into the correct periods
predictions = []
actuals = []
dates = []

# number of days of data to be used
min_train_size = 252

# starts at day 252 and runs until the end of gsk dataframe, i.e everything before i is the past data and i itself is the prediction
# essentially trains from day 1 to day 252, analysing the changes in metrics outlined earlier, then predicts what happens on day 253,
# then the timeframe for historical data increases by 1 day, so now the model has data from day 1 to day 53, and repeats the process
# until the end of the dataset.
for i in range(min_train_size, len(gsk)):
    # slices the features from gsk dataset from the very beginning up to, but not including, i.
    x_train = x.iloc[:i]
    # slices the target column from gsk dataset for the same period
    y_train = y.iloc[:i]

    # selects the row from features at index i, isolating the data for "day"/the day we want to predict
    x_test = x.iloc[i:i+1]
    # selects the actual result at index i, setting aside the real value so it can be used to compare against
    y_test = y.iloc[i]

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # creates a new random forest model with 200 trees with a max of 4 depth for this specific week
    model = RandomForestRegressor(
        # number of trees
        n_estimators=200,
        # depth of the trees
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    # calls training algorithm from earlier to study the past data to find patterns
    model.fit(x_train_scaled, y_train)
    # uses what's learned from past data to predict the value next day
    pred = model.predict(x_test_scaled)[0]

    # adds the models prediction back into a list
    predictions.append(float(pred))
    # adds the real result to a list
    actuals.append(float(y_test))
    # adds the date associated with i to a list
    dates.append(gsk.index[i])


# dataFrame to hold results
results = pd.DataFrame({
    "actual_log_return": actuals,
    "pred_log_return": predictions
}, index=dates)

# convert log returns back to normal returns, leaving us with profit/loss
results["actual_return"] = np.exp(results["actual_log_return"]) - 1
results["pred_return"] = np.exp(results["pred_log_return"]) - 1

# gets the actual closing prices for the dates we predicted
# aligns these dates with the gsk["Close"] list
results["Actual_Close"] = gsk.loc[results.index, "Close"]

# Resets the starting point used for learning each day, model doesn't need to know data
# from 3 months ago
results["Yesterday_Actual_Close"] = gsk["Close"].shift(1).loc[results.index]

# takes yesterday's real closing price and add the predicted return for today
# meaning as the model now refreshes in a sense every day, if it messes up, on the following
# day it will start fresh and recover
# Formula: Yesterday's Real Price * (1 + Predicted Return)
results["Predicted_Price"] = results["Yesterday_Actual_Close"] * (1 + results["pred_return"])

plt.figure(figsize=(12, 6))

results.index = pd.to_datetime(dates)
# plot the Actual Price (Blue)
plt.plot(results["Actual_Close"], label="Actual Price", color="blue", alpha=0.5)

# plot the ML Prediction (Orange), where it resets every day
plt.plot(results["Predicted_Price"], label="ML Predicted Next Day Price", color="orange", alpha=0.9, linewidth=1.5)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gcf().autofmt_xdate()

plt.title("GSK ML Price Prediction (One-Day Ahead Forecast)")
plt.xlabel("Date")
plt.ylabel("Price (GBX)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()