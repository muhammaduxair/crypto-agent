from data_fetcher import DataFetchingAgent
from prediction_engine import PredictionEngine
from llm_predictor import LLMPredictor
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()


def parse_price(price_str):
    try:
        return float(price_str.replace(',', ''))
    except ValueError:
        return None


def main():
    # Initialize agents
    data_agent = DataFetchingAgent()
    prediction_engine = PredictionEngine()
    llm_predictor = LLMPredictor()

    # Set parameters
    coin_id = 'bitcoin'
    vs_currency = 'usd'
    days = 30

    try:
        # Fetch data
        df = data_agent.fetch_data(coin_id, vs_currency, days)

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
        technical_indicators = X.tail(3).to_string()
        llm_prediction = llm_predictor.predict(
            market_data, technical_indicators, current_price)

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
            df, ml_prediction, max(ml_probabilities))

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
        prediction_table.add_row("  Estimated Price Range (24h):", "")
        prediction_table.add_row("    Lower Bound", f"${ml_lower_bound:,.2f}")
        prediction_table.add_row("    Upper Bound", f"${ml_upper_bound:,.2f}")
        prediction_table.add_row("Large Language Model:", "")
        prediction_table.add_row("  Prediction", llm_prediction_text)
        prediction_table.add_row("  Confidence", llm_conf)
        prediction_table.add_row("  Estimated Price Range (24h):", "")
        if llm_lower_bound is not None and llm_upper_bound is not None:
            prediction_table.add_row("    Lower Bound", f"${
                                     llm_lower_bound:,.2f}")
            prediction_table.add_row("    Upper Bound", f"${
                                     llm_upper_bound:,.2f}")
        else:
            prediction_table.add_row("    Price Range", "Unable to parse")
        console.print(Panel(prediction_table,
                      title="Predictions for the Next 24 Hours", expand=False))

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
