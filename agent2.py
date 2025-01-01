import ccxt
import pandas as pd
import pandas_ta as ta
import plotext as plt
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import print
from groq import Groq
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tweepy
from web3 import Web3
import os
import dotenv
import time

# Load environment variables
dotenv.load_dotenv()
console = Console()


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


class EnhancedCryptoPredictionAgent:
    def __init__(self):
        self.setup_exchange()
        self.setup_ai_client()
        self.setup_social_media()
        self.setup_blockchain()
        self.setup_sentiment_model()
        self.df = None
        self.future_df = None
        self.ticker = None
        self.expected_price = None
        self.sentiment = None
        self.scaler = MinMaxScaler()

    def setup_exchange(self):
        """Initialize exchange with rate limiting and error handling"""
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1200,
                'timeout': 30000,
            })
        except Exception as e:
            raise ConnectionError(f"Failed to initialize exchange: {str(e)}")

    def setup_ai_client(self):
        """Initialize AI client with API key validation"""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)

    def setup_social_media(self):
        """Initialize social media API clients"""
        twitter_api_key = os.environ.get("TWITTER_API_KEY")
        twitter_api_secret = os.environ.get("TWITTER_API_SECRET")
        twitter_access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
        twitter_access_secret = os.environ.get("TWITTER_ACCESS_SECRET")

        if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_secret]):
            auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
            auth.set_access_token(twitter_access_token, twitter_access_secret)
            self.twitter_client = tweepy.API(auth)
        else:
            console.print(
                "[yellow]Twitter API credentials not found. Social sentiment will be limited.[/yellow]")
            self.twitter_client = None

    def setup_blockchain(self):
        """Initialize Web3 connection for on-chain metrics"""
        try:
            infura_url = os.environ.get("INFURA_URL")
            if infura_url:
                self.web3 = Web3(Web3.HTTPProvider(infura_url))
            else:
                console.print(
                    "[yellow]Infura URL not found. On-chain metrics will be limited.[/yellow]")
                self.web3 = None
        except Exception as e:
            console.print(f"[yellow]Error setting up Web3: {str(e)}[/yellow]")
            self.web3 = None

    def setup_sentiment_model(self):
        """Initialize sentiment analysis model"""
        try:
            model_name = "finiteautomata/bertweet-base-sentiment-analysis"
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                model_name)
        except Exception as e:
            console.print(f"[yellow]Error setting up sentiment model: {
                          str(e)}[/yellow]")
            self.sentiment_model = None

    def get_social_sentiment(self, symbol: str) -> float:
        """Analyze social media sentiment"""
        if not self.twitter_client or not self.sentiment_model:
            return 0.0

        try:
            # Get recent tweets about the cryptocurrency
            search_query = f"#{symbol.split('/')[0].lower()}"
            tweets = self.twitter_client.search_tweets(
                q=search_query, lang="en", count=100)

            sentiments = []
            for tweet in tweets:
                inputs = self.sentiment_tokenizer(
                    tweet.text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.sentiment_model(**inputs)
                sentiment_score = torch.softmax(outputs.logits, dim=1)
                # Positive sentiment score
                sentiments.append(sentiment_score[0][1].item())

            return np.mean(sentiments) if sentiments else 0.0
        except Exception as e:
            console.print(f"[yellow]Error analyzing social sentiment: {
                          str(e)}[/yellow]")
            return 0.0

    def get_onchain_metrics(self, symbol: str) -> dict:
        """Fetch on-chain metrics"""
        if not self.web3:
            return {}

        try:
            # Example: Get basic network metrics for ethereum-based tokens
            metrics = {
                'gas_price': self.web3.eth.gas_price,
                'block_number': self.web3.eth.block_number,
                'network_hashrate': self.web3.eth.hashrate if hasattr(self.web3.eth, 'hashrate') else None
            }
            return metrics
        except Exception as e:
            console.print(
                f"[yellow]Error fetching on-chain metrics: {str(e)}[/yellow]")
            return {}

    def calculate_advanced_indicators(self):
        """Calculate advanced technical indicators"""
        if self.df is None or self.df.empty:
            raise ValueError("No data available for technical analysis")

        try:
            # Volatility indicators
            self.df['ATR'] = ta.atr(
                self.df['high'], self.df['low'], self.df['close'])
            self.df['BB_upper'], self.df['BB_middle'], self.df['BB_lower'] = ta.bbands(
                self.df['close'])

            # Momentum indicators
            self.df['MFI'] = ta.mfi(
                self.df['high'], self.df['low'], self.df['close'], self.df['volume'])
            self.df['CMF'] = ta.cmf(
                self.df['high'], self.df['low'], self.df['close'], self.df['volume'])

            # Trend indicators
            self.df['ADX'] = ta.adx(
                self.df['high'], self.df['low'], self.df['close'])
            self.df['VWAP'] = ta.vwap(
                self.df['high'], self.df['low'], self.df['close'], self.df['volume'])

            # Fill NaN values
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')

        except Exception as e:
            raise Exception(f"Error calculating advanced indicators: {str(e)}")

    def prepare_features(self):
        """Prepare features for model training"""
        feature_columns = [
            'close', 'volume', 'RSI', 'MACD', 'ATR', 'MFI', 'CMF', 'ADX',
            'BB_upper', 'BB_lower', 'VWAP'
        ]

        # Add social and on-chain features
        self.df['social_sentiment'] = self.get_social_sentiment(self.symbol)
        onchain_metrics = self.get_onchain_metrics(self.symbol)
        for key, value in onchain_metrics.items():
            self.df[f'onchain_{key}'] = value
            feature_columns.append(f'onchain_{key}')

        # Scale features
        self.features_scaled = self.scaler.fit_transform(
            self.df[feature_columns])
        return self.features_scaled

    def train_ensemble_model(self):
        """Train ensemble of models for prediction"""
        X = self.prepare_features()
        y = self.df['close'].values

        # Prepare sequences for LSTM
        sequence_length = 10
        X_lstm = []
        y_lstm = []

        for i in range(len(X) - sequence_length):
            X_lstm.append(X[i:(i + sequence_length)])
            y_lstm.append(y[i + sequence_length])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        # Train LSTM model
        lstm_model = LSTMPredictor(
            input_dim=X.shape[1],
            hidden_dim=64,
            num_layers=2
        )

        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X, y)

        return lstm_model, rf_model

    def predict_future_prices(self):
        """Predict future prices using ensemble approach"""
        if self.df is None or self.df.empty:
            raise ValueError("No data available for prediction")

        console.print("\n[bold green]Predicting future prices...[/bold green]")

        try:
            # Train models
            lstm_model, rf_model = self.train_ensemble_model()

            # Make predictions
            last_sequence = self.features_scaled[-10:]
            lstm_pred = lstm_model(
                torch.FloatTensor(last_sequence).unsqueeze(0))
            rf_pred = rf_model.predict(self.features_scaled[-1:])

            # Ensemble predictions (weighted average)
            final_prediction = (0.6 * lstm_pred.item() + 0.4 * rf_pred[0])

            # Store predictions
            self.expected_price = final_prediction

            # Get LLM analysis
            self._get_llm_analysis()

        except Exception as e:
            raise Exception(f"Error in price prediction: {str(e)}")

    def get_user_inputs(self):
        """Get and validate user inputs"""
        console.print(
            "\n[bold cyan]🤖 Welcome to Crypto Prediction Agent![/bold cyan]\n")

        while True:
            self.symbol = Prompt.ask("[yellow]Enter cryptocurrency symbol[/yellow]",
                                     default="BTC/USDT")
            if self.validate_symbol(self.symbol):
                break
            console.print("[red]Invalid symbol. Please try again.[/red]")

        valid_timeframes = ["1h", "4h", "1d", "1w"]
        self.timeframe = Prompt.ask(
            "[yellow]Enter timeframe for historical data[/yellow]",
            choices=valid_timeframes,
            default="1d"
        )

        while True:
            try:
                self.prediction_timeframe = int(Prompt.ask(
                    "[yellow]Enter prediction timeframe (days)[/yellow]",
                    default="30"
                ))
                if 1 <= self.prediction_timeframe <= 365:
                    break
                console.print(
                    "[red]Please enter a value between 1 and 365 days.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")

    def fetch_market_data(self):
        """Fetch market data with retry mechanism and error handling"""
        console.print("\n[bold green]Fetching market data...[/bold green]")

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=self.timeframe,
                    limit=1000
                )

                if not ohlcv:
                    raise ValueError("No data received from exchange")

                self.df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open',
                             'high', 'low', 'close', 'volume']
                )
                self.df['timestamp'] = pd.to_datetime(
                    self.df['timestamp'], unit='ms')

                # Fetch current ticker
                self.ticker = self.exchange.fetch_ticker(self.symbol)
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch market data after {
                                    max_retries} attempts: {str(e)}")
                console.print(f"[yellow]Retry {
                              attempt + 1}/{max_retries}...[/yellow]")
                time.sleep(retry_delay)

    def _get_llm_analysis(self):
        """Get detailed price prediction analysis from LLM"""
        try:
            # Calculate key metrics
            current_price = self.df['close'].iloc[-1]
            predicted_prices = self.future_df['predicted_price'].values
            final_price = predicted_prices[-1]
            price_change = ((final_price - current_price) /
                            current_price) * 100

            # Prepare technical indicators
            last_rsi = self.df['RSI'].iloc[-1]
            last_macd = self.df['MACD'].iloc[-1]

            prompt = f"""
            Analyze the following cryptocurrency data and provide a detailed prediction:

            Current Price: ${current_price:.2f}
            Predicted Price in {self.prediction_timeframe} days: ${final_price:.2f}
            Price Change: {price_change:.2f}%
            Technical Indicators:
            - RSI: {last_rsi:.2f}
            - MACD: {last_macd:.2f}

            Provide a concise analysis in this exact format:
            Expected Price: <price>
            Sentiment: <Bullish/Bearish>
            Reasoning: <1-2 sentences explaining why>
            """

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.2,
                max_tokens=200
            )

            analysis = response.choices[0].message.content
            self._parse_llm_analysis(analysis)

        except Exception as e:
            raise Exception(f"Error in LLM analysis: {str(e)}")

    def _parse_llm_analysis(self, analysis: str):
        """Parse LLM analysis with improved error handling"""
        try:
            lines = analysis.split('\n')

            # Extract expected price
            price_line = next(
                (line for line in lines if line.startswith('Expected Price:')), None)
            if price_line:
                price_str = price_line.split(
                    ':')[1].strip().replace('$', '').replace(',', '')
                self.expected_price = float(price_str)

            # Extract sentiment
            sentiment_line = next(
                (line for line in lines if line.startswith('Sentiment:')), None)
            if sentiment_line:
                self.sentiment = sentiment_line.split(':')[1].strip()

            # Extract reasoning
            reasoning_line = next(
                (line for line in lines if line.startswith('Reasoning:')), None)
            if reasoning_line:
                self.prediction_reasoning = reasoning_line.split(':')[
                    1].strip()

            if not all([self.expected_price, self.sentiment, self.prediction_reasoning]):
                raise ValueError("Incomplete analysis from LLM")

        except Exception as e:
            raise ValueError(f"Error parsing LLM analysis: {str(e)}")

    def display_future_prices(self):
        """Display future price predictions with improved formatting and reasoning"""
        if self.future_df is None or self.expected_price is None:
            raise ValueError("No prediction data available")

        console.print("\n[bold blue]Predicted Future Prices[/bold blue]")

        # Create prediction table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row(
            "Current Price",
            f"${self.df['close'].iloc[-1]:.2f}"
        )
        table.add_row(
            "Expected Price",
            f"${self.expected_price:.2f}"
        )
        table.add_row(
            "Prediction Range",
            f"${self.future_df['lower_bound'].mean(
            ):.2f} - ${self.future_df['upper_bound'].mean():.2f}"
        )
        table.add_row(
            "Market Sentiment",
            f"[{'green' if self.sentiment.lower() == 'bullish' else 'red'}]{
                self.sentiment}[/]"
        )

        console.print(table)

        # Display reasoning
        if hasattr(self, 'prediction_reasoning'):
            console.print("\n[bold blue]Analysis Reasoning:[/bold blue]")
            console.print(f"[italic]{self.prediction_reasoning}[/italic]")

    def plot_price_chart(self):
        """Plot price chart with confidence intervals"""
        if self.df is None or self.future_df is None:
            raise ValueError("No data available for plotting")

        console.print("\n[bold blue]Price Chart[/bold blue]")

        try:
            plt.clear_figure()

            # Plot historical prices
            plt.plot(self.df['close'].values,
                     label="Historical Price", color="blue")

            # Plot technical indicators
            if not self.df['EMA_20'].isna().all():
                plt.plot(self.df['EMA_20'].values,
                         label="20 EMA", color="green")
            if not self.df['SMA_50'].isna().all():
                plt.plot(self.df['SMA_50'].values, label="50 SMA", color="red")

            # Plot predictions with confidence intervals
            predictions = self.future_df['predicted_price'].values
            plt.plot(range(len(self.df), len(self.df) + len(predictions)),
                     predictions, label="Predicted", color="yellow")

            plt.title(f"{self.symbol} Price Chart with Predictions")
            plt.show()

        except Exception as e:
            raise Exception(f"Error plotting chart: {str(e)}")

    def display_market_info(self):
        """Display market information with data validation"""
        if not self.ticker:
            raise ValueError("No market data available")

        console.print("\n[bold blue]Current Market Information[/bold blue]")

        try:
            market_table = Table(show_header=True, header_style="bold magenta")
            market_table.add_column("Metric")
            market_table.add_column("Value")

            market_table.add_row("Current Price", f"${
                                 self.ticker['last']:.2f}")
            market_table.add_row("24h High", f"${self.ticker['high']:.2f}")
            market_table.add_row("24h Low", f"${self.ticker['low']:.2f}")
            market_table.add_row(
                "24h Volume", f"${self.ticker['quoteVolume']:,.2f}")
            market_table.add_row(
                "24h Change", f"{self.ticker['percentage']:.2f}%")

            console.print(market_table)

        except Exception as e:
            raise Exception(f"Error displaying market info: {str(e)}")

    def run(self):
        """Main execution flow with comprehensive error handling"""
        try:
            self.get_user_inputs()
            self.fetch_market_data()
            self.calculate_advanced_indicators()
            self.predict_future_prices()

            # Display information
            self.display_market_info()
            self.plot_price_chart()
            self.display_future_prices()

        except KeyboardInterrupt:
            console.print(
                "\n[bold yellow]Program terminated by user[/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
            console.print(
                "\n[bold yellow]Please try again or contact support if the issue persists.[/bold yellow]")


if __name__ == "__main__":
    agent = EnhancedCryptoPredictionAgent()
    agent.run()
