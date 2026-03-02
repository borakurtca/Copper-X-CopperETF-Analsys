import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
# Senin klasöründeki dosya ismine göre güncellendi
from news_sentiment import fetch_copper_news

# 1. Data process
df = pd.read_csv("processed_data_enriched.csv", index_col=0)
df['Copper_Mom'] = df['HG=F']
df['Market_Mom'] = df['^GSPC']
df['Vol_Shock'] = df['COPX_Purified'].rolling(window=5).std()
df['Target'] = df['COPX_Purified'].shift(-1)

df_ml = df[['Copper_Mom', 'Market_Mom', 'Vol_Shock', 'Target']].dropna()
X = df_ml.drop(columns=['Target'])
y = df_ml['Target']

# Test train split (%20)
split_idx = int(len(X) * 0.8)
X_test, y_test = X[split_idx:], y[split_idx:]

# 2. XGBoost
best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.7}
model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
model.fit(X[:split_idx], y[:split_idx])
preds = model.predict(X_test)

# 3. News analsys
news_df = fetch_copper_news()
current_sentiment = news_df['sentiment'].mean() if not news_df.empty else 0


# 4. Trend and news filters
results = pd.DataFrame({'Actual': y_test.values, 'Tech_Pred': preds}, index=y_test.index)
results['Final_Signal'] = np.sign(results['Tech_Pred'])

results['Market_Trend'] = results['Actual'].rolling(window=20).mean()
results.loc[(results['Market_Trend'] > 0) & (results['Final_Signal'] < 0), 'Final_Signal'] = 0
if current_sentiment < -0.10:
    results.loc[results['Final_Signal'] > 0, 'Final_Signal'] = 0

# test
active_trades = results[results['Final_Signal'] != 0]
new_accuracy = (np.sign(active_trades['Final_Signal']) == np.sign(active_trades['Actual'])).mean() * 100
results['Strategy_Return'] = results['Final_Signal'] * results['Actual']
results['Final_Wealth'] = 10000 * (1 + results['Strategy_Return']).cumprod()

print(f"\n--- RECOVERY REPORT ---")
print(f"New Filtered Accuracy: %{new_accuracy:.2f}")
print(f"Final Wealth: ${results['Final_Wealth'].iloc[-1]:.2f}")

# Casa test (10,000 USD)
initial_capital = 10000
results['Strategy_Return'] = results['Final_Signal'] * results['Actual']
results['Final_Wealth'] = initial_capital * (1 + results['Strategy_Return']).cumprod()

print(f"\n--- INTEGRATED VALIDATION REPORT ---")
print(f"Old Technical Accuracy (XGBoost): %47.62")
print(f"New Integrated Accuracy (RSS Filter): %{new_accuracy:.2f}")
print(f"Current RSS Sentiment Score: {current_sentiment:.2f}")
print(f"Final Wealth (10k Starting): ${results['Final_Wealth'].iloc[-1]:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(results['Final_Wealth'], color='blue', label='Integrated Strategy Wealth')
plt.axhline(y=10000, color='red', linestyle='--', label='Initial Capital')
plt.title("Backtest: 10,000 USD with RSS & Strategy Validation")
plt.legend()
plt.show()