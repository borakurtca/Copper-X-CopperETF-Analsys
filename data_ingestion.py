import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time


def get_enriched_data():
    print("--- Fetching Enriched Market Data... ---")
    # HG=F: Copper, COPX: Miners, DX-Y.NYB: Dollar, CL=F: Oil, ^GSPC: S&P 500, SI=F: Silver
    tickers = ['HG=F', 'COPX', 'DX-Y.NYB', 'CL=F', '^GSPC', 'SI=F']

    # Attempting to download data with retry logic
    data = None
    for i in range(3):
        try:
            data = yf.download(tickers, start="2021-01-01", auto_adjust=True)['Close']
            if not data.empty: break
        except Exception as e:
            print(f"Retry {i + 1} due to: {e}")
            time.sleep(2)

    data = data.dropna()
    returns = np.log(data / data.shift(1)).dropna()

    # --- ADVANCED PURIFICATION (Market Neutral Strategy) ---
    # We remove DXY, Oil, AND S&P 500 (Market Index) from COPX
    # This isolates the "Idiosyncratic Copper Signal"
    X_macro = returns[['DX-Y.NYB', 'CL=F', '^GSPC']]
    y_target = returns['COPX']

    cleaner_model = LinearRegression()
    cleaner_model.fit(X_macro, y_target)

    # Residuals = Purified Signal
    returns['COPX_Purified'] = y_target - cleaner_model.predict(X_macro)

    print("--- Feature Enrichment Complete: Market Beta Removed ---")
    return returns


if __name__ == "__main__":
    df = get_enriched_data()
    df.to_csv("processed_data_enriched.csv")
    print("Saved to processed_data_enriched.csv")