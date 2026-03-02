import feedparser
import pandas as pd
from textblob import TextBlob
from datetime import datetime


def fetch_copper_news():
    rss_urls = [
        "https://www.investing.com/rss/news_253.rss",
        "https://www.reutersagency.com/feed/?best-topics=commodities&post_type=best",
        "https://www.mining.com/feed/",
        "https://www.aljazeera.com/xml/rss/all.xml"
    ]

    news_data = []
    keywords = ['war', 'conflict', 'geopolitical', 'copper', 'metal', 'mining', 'attack', 'tension']

    print("--- Fetching Current News for Validation... ---")
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if any(word in entry.title.lower() for word in keywords):
                    analysis = TextBlob(entry.title)
                    news_data.append({
                        'title': entry.title,
                        'sentiment': analysis.sentiment.polarity
                    })
        except:
            continue

    return pd.DataFrame(news_data)


if __name__ == "__main__":
    df = fetch_copper_news()
    print(df.head())