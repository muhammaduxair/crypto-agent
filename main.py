import warnings
from data_fetcher import DataFetchingAgent
from prediction_engine import PredictionEngine
from llm_predictor import LLMPredictor
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

console = Console()


def get_user_input(prompt, default_value=None):
    while True:
        user_input = console.input(f"{prompt} [{default_value or ''}] : ")
        if not user_input:
            if default_value is not None:
                return default_value
            else:
                continue
        try:
            return int(user_input)
        except ValueError:
            console.print(
                "[bold red]Error:[/bold red] Invalid input. Please enter a number.")


def validate_coin_id(coin_id):
    try:
        data_agent = DataFetchingAgent()
        data_agent.cg.get_coin_by_id(coin_id)
        return True
    except Exception:
        return False


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

        metrics = prediction_engine.train(X, y)

        # Make predictions
        latest_data = X.iloc[-1].to_frame().T
        ml_prediction, ml_confidence, ml_metrics = prediction_engine.predict(
            latest_data)

        current_price = df['price'].iloc[-1]

        # Reduce the amount of data sent to LLM predictor
        market_data = df.tail(3)[['price', 'high', 'low', 'volume']].to_dict()
        technical_indicators = X.tail(1).to_dict()

        llm_prediction = llm_predictor.predict(
            market_data, technical_indicators, current_price, prediction_days)

        # Calculate ML model price range
        ml_lower_bound, ml_upper_bound = prediction_engine.calculate_price_range(
            df, ml_prediction, ml_confidence, prediction_days)

        # Create rich text for predictions
        ml_prediction_text = Text("Bullish", style="bold green") if ml_prediction == 1 else Text(
            "Bearish", style="bold red")
        llm_prediction_text = Text(llm_prediction.get('prediction', 'N/A'), style="bold green") if llm_prediction.get(
            'prediction', '').lower() == "bullish" else Text(llm_prediction.get('prediction', 'N/A'), style="bold red")

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
        prediction_table.add_row("  Confidence", f"{ml_confidence:.2f}")
        prediction_table.add_row(f"  Estimated Price Range ({prediction_days} day{
                                 's' if prediction_days > 1 else ''}):", "")
        prediction_table.add_row("    Lower Bound", f"${ml_lower_bound:,.2f}")
        prediction_table.add_row("    Upper Bound", f"${ml_upper_bound:,.2f}")
        prediction_table.add_row("Large Language Model:", "")
        prediction_table.add_row("  Prediction", llm_prediction_text)
        prediction_table.add_row(
            "  Confidence", f"{llm_prediction.get('confidence', 'N/A')}")
        prediction_table.add_row(f"  Estimated Price Range ({prediction_days} day{
                                 's' if prediction_days > 1 else ''}):", "")
        if 'price_range' in llm_prediction and 'lower' in llm_prediction['price_range'] and 'upper' in llm_prediction['price_range']:
            prediction_table.add_row("    Lower Bound", f"${
                                     llm_prediction['price_range']['lower']:,.2f}")
            prediction_table.add_row("    Upper Bound", f"${
                                     llm_prediction['price_range']['upper']:,.2f}")
        else:
            prediction_table.add_row("    Price Range", "Unable to calculate")
        console.print(Panel(prediction_table, title=f"Predictions for the Next {
                      prediction_days} Day{'s' if prediction_days > 1 else ''}", expand=False))

        if 'reasoning' in llm_prediction:
            console.print(
                Panel(Text(llm_prediction['reasoning']), title="LLM Reasoning", expand=False))
        if 'key_factors' in llm_prediction:
            console.print(Panel(Text('\n'.join(
                llm_prediction['key_factors'])), title="Key Factors (LLM)", expand=False))

        feature_importance = prediction_engine.get_model_diagnostics()[
            'feature_importance']
        feature_table = Table(show_header=False, box=None)
        for _, row in feature_importance.head().iterrows():
            feature_table.add_row(row['feature'], f"{row['importance']:.4f}")
        console.print(Panel(
            feature_table, title="Top 5 Most Important Features (ML Model)", expand=False))

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
