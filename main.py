from data_fetcher import DataFetchingAgent
from prediction_engine import PredictionEngine
from llm_predictor import LLMPredictor
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()
cg = CoinGeckoAPI()


def parse_price(price_str):
    try:
        return float(price_str.replace(',', ''))
    except ValueError:
        return None


def validate_coin_id(coin_id):
    try:
        cg.get_coin_by_id(coin_id)
        return True
    except Exception:
        return False


def get_user_input(prompt, default, type_func=int):
    while True:
        user_input = console.input(
            f"[bold cyan]{prompt} (default: {default}): [/bold cyan]")
        if user_input == "":
            return default
        try:
            return type_func(user_input)
        except ValueError:
            console.print(f"[bold red]Invalid input. Please enter a valid {
                          type_func.__name__}.[/bold red]")


def main():
    # Initialize agents
    data_agent = DataFetchingAgent()
    prediction_engine = PredictionEngine()
    llm_predictor = LLMPredictor()

    # Prompt user for inputs
    coin_id = console.input(
        "[bold cyan]Enter the coin ID (e.g., bitcoin, ethereum): [/bold cyan]")

    # Validate coin ID
    if not validate_coin_id(coin_id):
        console.print(f"[bold red]Error:[/bold red] Invalid coin ID '{
                      coin_id}'. Please check the ID and try again.")
        return

    historical_days = get_user_input(
        "Enter the number of days of historical data to fetch", 30)
    prediction_days = get_user_input(
        "Enter the number of days for prediction", 1)

    vs_currency = 'usd'

    try:
        # Fetch data
        console.print(
            f"Fetching {historical_days} days of historical data for {coin_id}...")
        df = data_agent.fetch_data(coin_id, vs_currency, historical_days)

        if df.empty:
            console.print(
                "[bold red]Error:[/bold red] No data fetched. Please check your internet connection and API status.")
            return

        # Prepare data and train the model
        X, y = prediction_engine.prepare_data(df)

        if X.empty or y.empty:
            console.print(
                "[bold red]Error:[/bold red] No valid data after preparation. Please check the data quality and preparation process.")
            return

        prediction_engine.train(X, y)

        # Make predictions
        latest_data = X.iloc[-1].to_frame().T
        ml_prediction, ml_probabilities = prediction_engine.predict(
            latest_data)

        current_price = df['price'].iloc[-1]
        market_data = df.tail(
            3)[['price', 'high', 'low', 'volume']].to_string()
        if 'yf_close' in df.columns and 'av_close' in df.columns:
            market_data += "\n" + \
                df.tail(3)[['yf_close', 'yf_volume',
                            'av_close', 'av_volume']].to_string()
        technical_indicators = X.tail(3).to_string()
        llm_prediction = llm_predictor.predict(
            market_data, technical_indicators, current_price, prediction_days)

        # Extract LLM prediction details
        llm_lines = llm_prediction.split('\n')
        llm_pred = next((line.split(
            ': ')[1] for line in llm_lines if line.startswith('Prediction:')), 'N/A')
        llm_conf = next((line.split(
            ': ')[1] for line in llm_lines if line.startswith('Confidence:')), 'N/A')
        llm_lower_bound = parse_price(next(
            (line.split('$')[1] for line in llm_lines if '- Lower Bound:' in line), 'N/A'))
        llm_upper_bound = parse_price(next(
            (line.split('$')[1] for line in llm_lines if '- Upper Bound:' in line), 'N/A'))
        llm_reasoning = '\n'.join(llm_lines[llm_lines.index(
            'Reasoning:') + 1:llm_lines.index('Key Factors:')])
        llm_key_factors = '\n'.join(
            llm_lines[llm_lines.index('Key Factors:') + 1:])

        # Calculate ML model price range
        ml_lower_bound, ml_upper_bound = prediction_engine.calculate_price_range(
            df, ml_prediction, max(ml_probabilities), prediction_days)

        # Create rich text for predictions
        ml_prediction_text = Text("Bullish", style="bold green") if ml_prediction == 1 else Text(
            "Bearish", style="bold red")
        llm_prediction_text = Text(llm_pred, style="bold green") if llm_pred.lower(
        ) == "bullish" else Text(llm_pred, style="bold red")

        # Print results
        console.print(Panel.fit(
            f"[bold]Crypto Prediction Report for {
                coin_id.capitalize()}[/bold]",
            border_style="bold",
            padding=(1, 1)
        ))

        price_table = Table(show_header=False, box=None)
        price_table.add_row("Current Price", f"${current_price:,.2f}")
        price_table.add_row("24h High", f"${df['high'].iloc[-1]:,.2f}")
        price_table.add_row("24h Low", f"${df['low'].iloc[-1]:,.2f}")
        console.print(Panel(price_table, title="Market Summary", expand=False))

        prediction_table = Table(show_header=False, box=None)
        prediction_table.add_row("Machine Learning Model:", "")
        prediction_table.add_row("  Prediction", ml_prediction_text)
        prediction_table.add_row(
            "  Confidence", f"{max(ml_probabilities):.2f}")
        prediction_table.add_row(f"  Estimated Price Range ({prediction_days} day{
                                 's' if prediction_days > 1 else ''}):", "")
        prediction_table.add_row("    Lower Bound", f"${ml_lower_bound:,.2f}")
        prediction_table.add_row("    Upper Bound", f"${ml_upper_bound:,.2f}")
        prediction_table.add_row("Large Language Model:", "")
        prediction_table.add_row("  Prediction", llm_prediction_text)
        prediction_table.add_row("  Confidence", llm_conf)
        prediction_table.add_row(f"  Estimated Price Range ({prediction_days} day{
                                 's' if prediction_days > 1 else ''}):", "")
        if llm_lower_bound is not None and llm_upper_bound is not None:
            prediction_table.add_row("    Lower Bound", f"${
                                     llm_lower_bound:,.2f}")
            prediction_table.add_row("    Upper Bound", f"${
                                     llm_upper_bound:,.2f}")
        else:
            prediction_table.add_row("    Price Range", "Unable to parse")
        console.print(Panel(prediction_table, title=f"Predictions for the Next {
                      prediction_days} Day{'s' if prediction_days > 1 else ''}", expand=False))

        console.print(Panel(Text(llm_reasoning),
                      title="LLM Reasoning", expand=False))
        console.print(Panel(Text(llm_key_factors),
                      title="Key Factors (LLM)", expand=False))

        feature_importance = prediction_engine.explain_prediction(X)
        feature_table = Table(show_header=False, box=None)
        for _, row in feature_importance.head().iterrows():
            feature_table.add_row(row['feature'], f"{row['importance']:.4f}")
        console.print(Panel(
            feature_table, title="Top 5 Most Important Features (ML Model)", expand=False))

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
