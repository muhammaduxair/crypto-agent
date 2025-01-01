import ccxt
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
from groq import Groq
import dotenv
import os

dotenv.load_dotenv()

console = Console()


def setup_ai_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=api_key)

# Function to get user input


def get_user_input():
    console.print(
        Panel("Cryptocurrency Prediction Agent", style="bold magenta"))
    pair = console.input(
        "[bold cyan]Enter cryptocurrency pair (e.g., BTC/USDT): [/bold cyan]")
    timeframe = console.input(
        "[bold cyan]Select timeframe for technical analysis (1d, 1w, 1m): [/bold cyan]")
    prediction_timeframe = console.input(
        "[bold cyan]Select prediction timeframe (1d, 1w, 1m, 3m): [/bold cyan]")
    return pair, timeframe, prediction_timeframe

# Function to fetch technical indicators using ccxt


def fetch_technical_indicators(exchange, symbol, timeframe):
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Fetching technical indicators...", total=100)

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=365)
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Calculate technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['Signal'] = calculate_macd(df['close'])

        # Add signals
        df['SMA_Signal'] = np.where(
            df['close'] > df['SMA_20'], 'Bullish', 'Bearish')
        df['EMA_Signal'] = np.where(
            df['close'] > df['EMA_20'], 'Bullish', 'Bearish')
        df['RSI_Signal'] = np.where(df['RSI'] > 70, 'Overbought (Bearish)', np.where(
            df['RSI'] < 30, 'Oversold (Bullish)', 'Neutral'))
        df['MACD_Signal'] = np.where(
            df['MACD'] > df['Signal'], 'Bullish', 'Bearish')

        progress.update(task, advance=100)

    return df

# Function to calculate RSI


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate MACD


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

# Function to fetch news sentiment


def fetch_news_sentiment(symbol):
    with Progress() as progress:
        task = progress.add_task("[cyan]Fetching news sentiment...", total=100)

        api_key = os.environ.get("NEWSAPI_KEY")
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={
            api_key}&language=en&sortBy=publishedAt&pageSize=10"
        response = requests.get(url)
        news_data = response.json()

        sentiments = []
        for article in news_data.get('articles', []):
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            sentiment = TextBlob(content).sentiment.polarity
            sentiments.append(sentiment)

        avg_sentiment = np.mean(sentiments) if sentiments else 0

        progress.update(task, advance=100)

    # Return average sentiment and top 5 articles
    return avg_sentiment, news_data.get('articles', [])[:5]

# Function to generate LLM predictions using Groq


def generate_llm_predictions(technical_data, news_sentiment, symbol, top_articles, prediction_timeframe):
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Generating LLM predictions...", total=100)

        # Prepare input context for the LLM
        context = f"""
        You are an expert cryptocurrency technical analyst with years of experience in market analysis and prediction. Your task is to analyze the provided data and generate a well-reasoned price prediction for {symbol}.

        Technical Indicators for {symbol}:
        1. Latest Close Price: {technical_data['close'].iloc[-1]:.2f}
        2. SMA_20: {technical_data['SMA_20'].iloc[-1]:.2f} (Signal: {technical_data['SMA_Signal'].iloc[-1]})
        3. EMA_20: {technical_data['EMA_20'].iloc[-1]:.2f} (Signal: {technical_data['EMA_Signal'].iloc[-1]})
        4. RSI: {technical_data['RSI'].iloc[-1]:.2f} (Signal: {technical_data['RSI_Signal'].iloc[-1]})
        5. MACD: {technical_data['MACD'].iloc[-1]:.2f} (Signal: {technical_data['MACD_Signal'].iloc[-1]})
        6. MACD Signal Line: {technical_data['Signal'].iloc[-1]:.2f}

        Additional Market Data:
        7. 24h Price Change: {((technical_data['close'].iloc[-1] / technical_data['close'].iloc[-2]) - 1) * 100:.2f}%
        8. 7-day Price Change: {((technical_data['close'].iloc[-1] / technical_data['close'].iloc[-7]) - 1) * 100:.2f}%
        9. 30-day Price Change: {((technical_data['close'].iloc[-1] / technical_data['close'].iloc[-30]) - 1) * 100:.2f}%
        10. 24h Trading Volume: {technical_data['volume'].iloc[-1]:,.0f}

        Market Sentiment:
        11. News Sentiment Score: {news_sentiment:.2f} (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive)

        Recent News Headlines:
        {' '.join([f"- {article['title']}" for article in top_articles])}

        Based on the above technical indicators, market data, news sentiment, and recent headlines, provide a comprehensive price prediction for {symbol} for the following timeframe: {prediction_timeframe}

        Your analysis should include:
        1. A price range prediction (low and high)
        2. A confidence level (low, medium, high) for your prediction
        3. A detailed explanation of your prediction, including:
           a. Key technical indicators influencing your decision
           b. Interpretation of market sentiment and news impact
           c. Potential catalysts or risks that could affect the price
           d. Any patterns or trends you've identified in the data
        4. Recommended trading strategy based on your analysis (e.g., buy, sell, hold)

        Remember to consider both bullish and bearish scenarios in your analysis. Your prediction should be well-reasoned, data-driven, and reflect your expertise in technical analysis and cryptocurrency markets.
        """

        client = setup_ai_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": context}],
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=200
        )

        analysis = response.choices[0].message.content

        progress.update(task, advance=100)

    return analysis

# Main function


def main():
    pair, timeframe, prediction_timeframe = get_user_input()

    exchange = ccxt.binance()

    technical_data = fetch_technical_indicators(exchange, pair, timeframe)
    news_sentiment, top_articles = fetch_news_sentiment(pair.split('/')[0])

    predictions = generate_llm_predictions(
        technical_data, news_sentiment, pair, top_articles, prediction_timeframe)

    # Display results
    console.print(Panel(f"Prediction Results for {pair}", style="bold green"))

    # Display technical indicators
    tech_table = Table(title="Technical Indicators")
    tech_table.add_column("Indicator", style="cyan")
    tech_table.add_column("Value", style="magenta")
    tech_table.add_column("Signal", style="yellow")

    tech_table.add_row("Close Price", f"{
                       technical_data['close'].iloc[-1]:.2f}", "-")
    tech_table.add_row("SMA_20", f"{
                       technical_data['SMA_20'].iloc[-1]:.2f}", technical_data['SMA_Signal'].iloc[-1])
    tech_table.add_row("EMA_20", f"{
                       technical_data['EMA_20'].iloc[-1]:.2f}", technical_data['EMA_Signal'].iloc[-1])
    tech_table.add_row("RSI", f"{
                       technical_data['RSI'].iloc[-1]:.2f}", technical_data['RSI_Signal'].iloc[-1])
    tech_table.add_row("MACD", f"{
                       technical_data['MACD'].iloc[-1]:.2f}", technical_data['MACD_Signal'].iloc[-1])
    tech_table.add_row("MACD Signal", f"{
                       technical_data['Signal'].iloc[-1]:.2f}", "-")

    console.print(tech_table)

    # Display news sentiment
    sentiment_panel = Panel(
        f"News Sentiment: {news_sentiment:.2f}", style="bold blue")
    console.print(sentiment_panel)

    # Display recent news headlines
    news_table = Table(title="Recent News Headlines")
    news_table.add_column("Title", style="cyan")
    news_table.add_column("Source", style="magenta")

    for article in top_articles:
        news_table.add_row(article['title'], article['source']['name'])

    console.print(news_table)

    # Display LLM predictions
    console.print(Panel(f"LLM Predictions for {
                  prediction_timeframe}", style="bold yellow"))
    console.print(predictions)


if __name__ == "__main__":
    main()
