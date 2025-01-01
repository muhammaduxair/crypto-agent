import ccxt
import pandas as pd
import pandas_ta as ta
import plotext as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from groq import Groq
from prophet import Prophet
import os
import dotenv
import time

# Load environment variables
dotenv.load_dotenv()

console = Console()


class CryptoPredictionAgent:
    def __init__(self):
        self.setup_exchange()
        self.setup_ai_client()
        self.df = None
        self.future_df = None
        self.ticker = None
        self.expected_price = None
        self.sentiment = None

    def setup_exchange(self):
        """Initialize exchange with rate limiting and error handling"""
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1200,  # Time in milliseconds between requests
                'timeout': 30000,   # 30 seconds timeout
            })
        except Exception as e:
            raise ConnectionError(f"Failed to initialize exchange: {str(e)}")

    def setup_ai_client(self):
        """Initialize AI client with API key validation"""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if the trading symbol exists"""
        try:
            markets = self.exchange.load_markets()
            return symbol in markets
        except Exception as e:
            console.print(f"[red]Error validating symbol: {str(e)}[/red]")
            return False

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

    def calculate_technical_indicators(self):
        """Calculate technical indicators with data validation"""
        if self.df is None or self.df.empty:
            raise ValueError("No data available for technical analysis")

        console.print(
            "\n[bold green]Calculating technical indicators...[/bold green]")

        try:
            # Handle NaN values
            self.df['close'] = self.df['close'].ffill()

            # Add technical indicators with error checking
            self.df['RSI'] = ta.rsi(self.df['close'])
            self.df['MACD'] = ta.macd(self.df['close'])['MACD_12_26_9']
            self.df['EMA_20'] = ta.ema(self.df['close'], length=20)
            self.df['SMA_50'] = ta.sma(self.df['close'], length=50)

            # Fill any remaining NaN values
            self.df = self.df.ffill().bfill()

        except Exception as e:
            raise Exception(
                f"Error calculating technical indicators: {str(e)}")

    def predict_future_prices(self):
        """Predict future prices with improved error handling and validation"""
        if self.df is None or self.df.empty:
            raise ValueError("No data available for prediction")

        console.print("\n[bold green]Predicting future prices...[/bold green]")

        try:
            # Prepare data for Prophet
            prophet_df = self.df[['timestamp', 'close']].rename(
                columns={'timestamp': 'ds', 'close': 'y'})

            # Clean and validate data
            prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
            prophet_df = prophet_df.dropna()

            if prophet_df.empty:
                raise ValueError("No valid data after cleaning")

            # Create and fit the model with appropriate parameters
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )
            model.fit(prophet_df)

            # Create future dataframe
            future = model.make_future_dataframe(
                periods=self.prediction_timeframe)
            forecast = model.predict(future)

            # Process prediction
            self.future_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(
                self.prediction_timeframe)
            self.future_df = self.future_df.rename(
                columns={
                    'ds': 'timestamp',
                    'yhat': 'predicted_price',
                    'yhat_lower': 'lower_bound',
                    'yhat_upper': 'upper_bound'
                }
            )

            # Get LLM analysis
            self._get_llm_analysis()

        except Exception as e:
            raise Exception(f"Error in price prediction: {str(e)}")

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
                temperature=0.7,
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
            self.calculate_technical_indicators()
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
    agent = CryptoPredictionAgent()
    agent.run()
