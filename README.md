A modular trading engine designed for directional forecasting of the COPX ETF using XGBoost. The system validates technical signals with real-time geopolitical sentiment analysis and trend-following safeguards (20-Day SMA) to mitigate false entries.
Performance: Improved directional accuracy from a 47.62% technical baseline to 58.54%.
Wealth Impact: Successfully grew a simulated $10k account to $12,396.70 during the backtest period.
Component / Bileşen,Description / Açıklama
Predictive Engine,XGBoost Regressor optimized for OHLCV and volatility shock data.
Sentiment Shield,Automated RSS scraping using NLP (TextBlob) to filter trades during high-risk events.
Trend Guardrail,Logic-based filter ensuring all trades align with the primary market trend (20-day SMA).
