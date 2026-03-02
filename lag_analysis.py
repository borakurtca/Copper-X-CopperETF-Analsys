import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned dataset we created in the previous step
# This file contains the "Purified" COPX (Macro effects removed)
try:
    df = pd.read_csv("processed_data.csv", index_col=0)
    print("--- Dataset loaded successfully ---")
except FileNotFoundError:
    print("Error: processed_data.csv not found. Run data_ingestion.py first!")

# 2. Calculate Cross-Correlation for different time lags
# We check from 0 (today) up to 10 days in the past
lags = range(0, 11)
correlations = []

for lag in lags:
    # We shift the Copper price 'lag' days forward to see its impact on COPX
    # .corr() measures how much these two columns move together
    correlation = df['HG=F'].shift(lag).corr(df['COPX_Purified'])
    correlations.append(correlation)

# 3. Visualization
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
sns.barplot(x=list(lags), y=correlations, palette="viridis")

# Formatting the plot
plt.title("Time-Lagged Correlation: Copper Futures vs. Purified COPX", fontsize=14)
plt.xlabel("Lag (Days)", fontsize=12)
plt.ylabel("Correlation Coefficient", fontsize=12)
plt.axhline(0, color='black', linewidth=1) # Baseline

# Identify the best lag day
best_lag = correlations.index(max(correlations))
plt.annotate(f'Peak Impact at Day {best_lag}',
             xy=(best_lag, max(correlations)),
             xytext=(best_lag+1, max(correlations)+0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

print(f"\n--- ANALYSIS RESULT ---")
print(f"The strongest relationship occurs at a {best_lag}-day lag.")
print(f"This means Copper price movements today best explain COPX returns {best_lag} days from now.")