import pandas as pd
import numpy as np
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
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.sia = SentimentIntensityAnalyzer()
        self.av_cc = CryptoCurrencies(
            key=os.environ.get('ALPHA_VANTAGE_API_KEY'))
        self.av_ti = TechIndicators(
            key=os.environ.get('ALPHA_VANTAGE_API_KEY'))

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
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                yf_data = yf.Ticker(yf_symbol).history(
                    start=start_date, end=end_date)
                if not yf_data.empty:
                    yf_data = yf_data.reindex(df.index, method='nearest')
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
                av_data, _ = self.av_cc.get_digital_currency_daily(
                    symbol=coin_id.upper(), market=vs_currency.upper())
                av_df = av_data.sort_index().last(f"{days}D")
                av_df.index = pd.to_datetime(av_df.index)
                av_df = av_df.reindex(df.index, method='nearest')
                df['av_close'] = av_df['4b. close (USD)'].astype(float)
                df['av_volume'] = av_df['5. volume'].astype(float)

                # Fetch BBands indicator
                bbands_data, _ = self.av_ti.get_bbands(symbol=f"{coin_id.upper()}{
                                                       vs_currency.upper()}", interval='daily', time_period=20)
                bbands_df = bbands_data.sort_index().last(f"{days}D")
                bbands_df.index = pd.to_datetime(bbands_df.index)
                bbands_df = bbands_df.reindex(df.index, method='nearest')
                df['bb_upper'] = bbands_df['Real Upper Band']
                df['bb_middle'] = bbands_df['Real Middle Band']
                df['bb_lower'] = bbands_df['Real Lower Band']
            except Exception as e:
                df['av_close'] = df['price']
                df['av_volume'] = df['volume']
                df['bb_upper'] = np.nan
                df['bb_middle'] = np.nan
                df['bb_lower'] = np.nan
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
