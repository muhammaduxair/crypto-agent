import pandas as pd
from pycoingecko import CoinGeckoAPI
import yfinance as yf
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.techindicators import TechIndicators
from ta import add_all_ta_features
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from rich.console import Console
import os
from datetime import datetime, timedelta

console = Console()

nltk.download('vader_lexicon', quiet=True)


class DataFetchingAgent:
    def __init__(self, alpha_vantage_key=None):
        self.cg = CoinGeckoAPI()
        self.sia = SentimentIntensityAnalyzer()
        self.alpha_vantage_key = alpha_vantage_key or os.environ.get(
            'ALPHA_VANTAGE_API_KEY')

        if self.alpha_vantage_key:
            self.av_cc = CryptoCurrencies(key=self.alpha_vantage_key)
            self.av_ti = TechIndicators(key=self.alpha_vantage_key)
        else:
            console.print(
                "[yellow]Warning:[/yellow] No Alpha Vantage API key provided.")

    def fetch_yfinance_data(self, coin_id, vs_currency, start_date, end_date):
        """Separate method for fetching yFinance data with better error handling"""
        try:
            # Try different symbol formats as backup options
            symbols = [
                f"{coin_id.upper()}-{vs_currency.upper()}",
                f"{coin_id.upper()}{vs_currency.upper()}=X",
                f"{coin_id.upper()}-{vs_currency.upper()}=X"
            ]

            for symbol in symbols:
                try:
                    yf_data = yf.download(
                        symbol, start=start_date, end=end_date, progress=False)
                    if not yf_data.empty:
                        return yf_data
                except:
                    continue

            raise Exception("No valid data found for any symbol format")

        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] yFinance data fetch failed: {str(e)}")
            return pd.DataFrame()

    def fetch_alpha_vantage_data(self, coin_id, vs_currency, days):
        """Separate method for fetching Alpha Vantage data with better error handling"""
        try:
            if not self.alpha_vantage_key:
                raise Exception("No Alpha Vantage API key provided")

            # Fetch daily cryptocurrency data
            av_data, _ = self.av_cc.get_digital_currency_daily(
                symbol=coin_id.upper(), market=vs_currency.upper())

            av_df = pd.DataFrame(av_data).astype(float)
            av_df.index = pd.to_datetime(av_df.index)

            # Get last N days of data
            av_df = av_df.last(f"{days}D")

            # Fetch Bollinger Bands
            bbands_data, _ = self.av_ti.get_bbands(
                symbol=f"{coin_id.upper()}{vs_currency.upper()}",
                interval='daily',
                time_period=20
            )

            bbands_df = pd.DataFrame(bbands_data).astype(float)
            bbands_df.index = pd.to_datetime(bbands_df.index)
            bbands_df = bbands_df.last(f"{days}D")

            return av_df, bbands_df

        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] Alpha Vantage data fetch failed: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def fetch_data(self, coin_id, vs_currency, days):
        try:
            # Fetch CoinGecko data
            cg_data = self.cg.get_coin_market_chart_by_id(
                id=coin_id, vs_currency=vs_currency, days=days)

            # Create main DataFrame
            df = pd.DataFrame(cg_data['prices'], columns=[
                              'timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add volume and OHLC data
            df['volume'] = [x[1] for x in cg_data['total_volumes']]
            df['high'] = df['price'].rolling(window=24).max()  # Daily high
            df['low'] = df['price'].rolling(window=24).min()   # Daily low
            df['open'] = df['price'].resample('D').first()     # Daily open
            df['close'] = df['price'].resample('D').last()     # Daily close

            # Fetch yFinance data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            yf_data = self.fetch_yfinance_data(
                coin_id, vs_currency, start_date, end_date)

            if not yf_data.empty:
                yf_data = yf_data.reindex(df.index, method='nearest')
                df['yf_close'] = yf_data['Close']
                df['yf_volume'] = yf_data['Volume']
            else:
                df['yf_close'] = df['price']
                df['yf_volume'] = df['volume']

            # Fetch Alpha Vantage data
            av_df, bbands_df = self.fetch_alpha_vantage_data(
                coin_id, vs_currency, days)

            if not av_df.empty:
                av_df = av_df.reindex(df.index, method='nearest')
                df['av_close'] = av_df['4b. close (USD)']
                df['av_volume'] = av_df['5. volume']
            else:
                df['av_close'] = df['price']
                df['av_volume'] = df['volume']

            if not bbands_df.empty:
                bbands_df = bbands_df.reindex(df.index, method='nearest')
                df['bb_upper'] = bbands_df['Real Upper Band']
                df['bb_middle'] = bbands_df['Real Middle Band']
                df['bb_lower'] = bbands_df['Real Lower Band']
            else:
                # Calculate Bollinger Bands manually if Alpha Vantage fails
                rolling_mean = df['price'].rolling(window=20).mean()
                rolling_std = df['price'].rolling(window=20).std()
                df['bb_upper'] = rolling_mean + (rolling_std * 2)
                df['bb_middle'] = rolling_mean
                df['bb_lower'] = rolling_mean - (rolling_std * 2)

            # Add sentiment analysis
            df['sentiment'] = self.fetch_sentiment(coin_id)

            # Add technical indicators
            df = add_all_ta_features(
                df,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=True
            )

            return df

        except Exception as e:
            console.print(
                f"[bold red]Error in main data fetch:[/bold red] {str(e)}")
            return pd.DataFrame()

    def fetch_sentiment(self, coin_id):
        try:
            coin_data = self.cg.get_coin_by_id(coin_id)
            description = coin_data.get('description', {}).get('en', '')

            # Also analyze recent tweets if available
            tweets = coin_data.get('twitter_feed', [])
            all_text = description + \
                ' '.join([tweet.get('text', '') for tweet in tweets])

            sentiment_score = self.sia.polarity_scores(all_text)['compound']
            return sentiment_score
        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] Sentiment analysis failed: {str(e)}")
            return 0
