import pandas as pd
import numpy as np
import feedparser
from textblob import TextBlob
from datetime import datetime


# 1. RSS NEWS ENGINE (Geopolitics & Copper)
def fetch_current_sentiment():
    rss_urls = [
        "https://www.investing.com/rss/news_253.rss",
        "https://www.reutersagency.com/feed/?best-topics=commodities&post_type=best",
        "https://www.mining.com/feed/",
        "https://www.aljazeera.com/xml/rss/all.xml"
    ]

    news_data = []
    keywords = ['war', 'conflict', 'geopolitical', 'copper', 'metal', 'mining', 'attack', 'tension']

    print("--- Scanning Global News & War Risks... ---")
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if any(word in entry.title.lower() for word in keywords):
                    analysis = TextBlob(entry.title)
                    news_data.append(analysis.sentiment.polarity)
        except:
            continue

    return np.mean(news_data) if news_data else 0


# 2. DATA VALIDATION
def run_daily_check():
    try:
        df = pd.read_csv("processed_data_enriched.csv", index_col=0)
        print("--- SUCCESS: Enriched Data Loaded ---")
    except FileNotFoundError:
        print("--- ERROR: processed_data_enriched.csv not found! ---")
        return

    # News scorring
    sentiment = fetch_current_sentiment()

    # Teknik Özellikler (Feature Importance grafiğine göre en etkili olanlar)
    last_vol = df['COPX_Purified'].rolling(window=5).std().iloc[-1]
    last_market_mom = df['^GSPC'].iloc[-1]

    print(f"\n--- DAILY VALIDATION REPORT ---")
    print(f"Current Global Sentiment Score: {sentiment:.2f}")
    print(f"Market Volatility (Vol_Shock): {last_vol:.4f}")

    # decison support
    print("\n--- DECISION SUPPORT ---")
    if sentiment < -0.10:
        print("ALERT: High Geopolitical Risk detected. AVOID LONG POSITIONS.")
    elif last_vol > 0.02:
        print("ADVICE: High Market Volatility. Stay cautious with XGBoost signals.")
    else:
        print("STATUS: Market conditions stable. You may follow Technical Signals.")


if __name__ == "__main__":
    run_daily_check()