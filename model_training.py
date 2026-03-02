import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Data process
df = pd.read_csv("processed_data_enriched.csv", index_col=0)
df['Copper_Mom'] = df['HG=F']
df['Market_Mom'] = df['^GSPC']
df['Vol_Shock'] = df['COPX_Purified'].rolling(window=5).std()
df['Target'] = df['COPX_Purified'].shift(-1)

df_ml = df[['Copper_Mom', 'Market_Mom', 'Vol_Shock', 'Target']].dropna()
X = df_ml.drop(columns=['Target'])
y = df_ml['Target']

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 2. Optimize parameter
best_params = {
    'colsample_bytree': 0.7,
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 50,
    'subsample': 0.7
}

model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 3. Stratagy and backtest
results = pd.DataFrame({'Actual': y_test.values, 'Pred': preds}, index=y_test.index)

threshold = 0.0005
results['Signal'] = 0
results.loc[results['Pred'] > threshold, 'Signal'] = 1
results.loc[results['Pred'] < -threshold, 'Signal'] = -1

results['Strategy_Return'] = results['Signal'] * results['Actual']

# cumulative income (10,000$)
initial_capital = 10000
results['Market_Wealth'] = initial_capital * (1 + results['Actual']).cumprod()
results['Strategy_Wealth'] = initial_capital * (1 + results['Strategy_Return']).cumprod()

# Report
final_strat = results['Strategy_Wealth'].iloc[-1]
final_mkt = results['Market_Wealth'].iloc[-1]

print(f"\n--- FINAL OPTIMIZED REPORT ---")
print(f"Final Strategy Wealth: ${final_strat:.2f}")
print(f"Market Passive Wealth: ${final_mkt:.2f}")
print(f"Win Rate (Direction): %{(np.sign(preds) == np.sign(y_test.values)).mean()*100:.2f}")

# vizilations
plt.figure(figsize=(12, 7))
plt.plot(results['Market_Wealth'], label='Market (Passive)', color='gray', alpha=0.5)
plt.plot(results['Strategy_Wealth'], label='Optimized Strategy', color='blue', linewidth=2)
plt.title("Final Backtest: Optimized XGBoost vs Market")
plt.legend()
plt.show()