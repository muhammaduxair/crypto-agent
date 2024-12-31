import pandas as pd
from pycoingecko import CoinGeckoAPI
import yfinance as yf
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from ta import add_all_ta_features
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from rich.console import Console
import os

console = Console()

nltk.download('vader_lexicon', quiet=True)


class DataFetchingAgent:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.sia = SentimentIntensityAnalyzer()
        self.av = CryptoCurrencies(key=os.environ.get('ALPHA_VANTAGE_API_KEY'))

    def fetch_data(self, coin_id, vs_currency, days):
        try:
            # Fetch data from CoinGecko
            cg_data = self.cg.get_coin_market_chart_by_id(
                id=coin_id, vs_currency=vs_currency, days=days)

            # Create DataFrame
            df = pd.DataFrame(cg_data['prices'], columns=[
                              'timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['volume'] = [x[1] for x in cg_data['total_volumes']]

            # Add high and low
            df['high'] = [x[1] for x in cg_data['prices']]
            df['low'] = [x[1] for x in cg_data['prices']]

            # Fetch yFinance data
            try:
                yf_symbol = f"{coin_id.upper()}-{vs_currency.upper()}"
                yf_data = yf.Ticker(yf_symbol).history(period=f"{days}d")
                if not yf_data.empty:
                    df['yf_close'] = yf_data['Close']
                    df['yf_volume'] = yf_data['Volume']
                else:
                    df['yf_close'] = df['price']
                    df['yf_volume'] = df['volume']
                    console.print(
                        "[yellow]Warning:[/yellow] No yFinance data available. Using CoinGecko data as substitute.")
            except Exception as e:
                df['yf_close'] = df['price']
                df['yf_volume'] = df['volume']
                console.print(
                    f"[yellow]Warning:[/yellow] Error fetching yFinance data: {e}")

            # Fetch Alpha Vantage data
            try:
                av_data, _ = self.av.get_digital_currency_daily(
                    symbol=coin_id.upper(), market=vs_currency.upper())
                av_df = pd.DataFrame.from_dict(av_data).T
                av_df.index = pd.to_datetime(av_df.index)
                av_df = av_df.sort_index().last(f"{days}D")
                df['av_close'] = av_df['4a. close (USD)'].astype(float)
                df['av_volume'] = av_df['5. volume'].astype(float)
            except Exception as e:
                df['av_close'] = df['price']
                df['av_volume'] = df['volume']
                console.print(
                    f"[yellow]Warning:[/yellow] Error fetching Alpha Vantage data: {e}")

            # Fetch and add sentiment data
            sentiment_data = self.fetch_sentiment(coin_id)
            df['sentiment'] = sentiment_data

            # Add technical indicators
            df = add_all_ta_features(
                df, open="price", high="high", low="low", close="price", volume="volume")

            return df

        except Exception as e:
            console.print(f"[bold red]Error fetching data:[/bold red] {e}")
            return pd.DataFrame()

    def fetch_sentiment(self, coin_id):
        try:
            coin_data = self.cg.get_coin_by_id(coin_id)
            text = coin_data.get('description', {}).get('en', '')
            sentiment_score = self.sia.polarity_scores(text)['compound']
            return sentiment_score
        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] Error fetching sentiment data: {e}")
            return 0
