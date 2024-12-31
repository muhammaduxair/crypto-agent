import requests
import pandas as pd
from ta import add_all_ta_features
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)


class DataFetchingAgent:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.sia = SentimentIntensityAnalyzer()

    def fetch_data(self, coin_id, vs_currency, days):
        # Fetch price data
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching data from API: {e}")
            return pd.DataFrame()

        print("API Response structure:")
        print(data.keys())

        # Create DataFrame
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Add high and low if available, otherwise use price
        if 'high' in data and 'low' in data:
            df['high'] = [x[1] for x in data['high']]
            df['low'] = [x[1] for x in data['low']]
        else:
            print(
                "Warning: 'high' and 'low' data not available. Using 'price' as a substitute.")
            df['high'] = df['price']
            df['low'] = df['price']

        print(f"Raw data shape: {df.shape}")
        print(df[['price', 'high', 'low']].head())
        print(f"Columns: {df.columns}")

        # Add volume data
        df['volume'] = [x[1] for x in data['total_volumes']]

        # Fetch and add sentiment data
        sentiment_data = self.fetch_sentiment(coin_id)
        df['sentiment'] = sentiment_data

        # Add technical indicators
        df = add_all_ta_features(
            df, open="price", high="high", low="low", close="price", volume="volume")

        print(f"Data shape after adding indicators: {df.shape}")
        print(df.head())
        print(f"Columns: {df.columns}")

        return df

    def fetch_sentiment(self, coin_id):
        url = f"{self.base_url}/coins/{coin_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching sentiment data: {e}")
            return 0

        # Extract relevant text for sentiment analysis
        text = data.get('description', {}).get('en', '')
        sentiment_score = self.sia.polarity_scores(text)['compound']

        return sentiment_score
