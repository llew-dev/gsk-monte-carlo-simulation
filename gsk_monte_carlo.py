import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gsk = pd.read_csv("gsk_daily.csv", index_col=0, parse_dates=True)
gsk["Close"] = pd.to_numeric(gsk["Close"], errors="coerce")
mu_daily = pd.read_csv("mu_daily.csv", index_col=0)

mu = mu_daily.loc["mu_daily"].values[0]
sigma = gsk["log_return"].std()
# number of days the simulation will use data from
days = 390
# number of simulations to be run
n_sims = 10000
# Starting price
S0 = float(gsk["Close"].iloc[-1])

# Time step
dt = 1

# randomly generated z scores
Z = np.random.normal(0, 1, (days, n_sims))

# paths matrix
paths = np.zeros((days + 1, n_sims))
paths[0] = S0

# Geometric Brownian Motion, repeated for every day
for t in range(1, days + 1):
    paths[t] = paths[t - 1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1]
    )


plt.figure(figsize=(10, 5))

# plotting 500 paths of the 10000 simulations ran
n_paths_to_plot = 500
random_indices = np.random.choice(n_sims, n_paths_to_plot, replace=False)

for i in random_indices:
    plt.plot(paths[:, i], alpha=0.3, linewidth=1)

# 2. plot average (mean) path
mean_path = paths.mean(axis=1)
plt.plot(mean_path,
         linewidth=2.5,
         label="Average Path (Predicted)"
)

# 3. current price reference line
plt.axhline(
    y=S0,
    linestyle="--",
    linewidth=1.5,
    label=f"Start Price: {round(S0)}p"
)

# labels & layout
plt.title(f"Monte Carlo Price Simulation for GSK {days} Day Horizon / ~ 1.5 years)")
plt.xlabel("Trading Days into Future")
plt.ylabel("Price (GBX)")
plt.gca().set_facecolor("black")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.tick_params(colors = "black")
plt.legend()
plt.show()

average_price = paths[-1].mean()
print(f"Expected price after {days} days: {average_price:.2f}p")

