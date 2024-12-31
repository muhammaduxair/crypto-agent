from data_fetcher import DataFetchingAgent
from prediction_engine import PredictionEngine
from llm_predictor import LLMPredictor
from dotenv import load_dotenv

# load the environment variables
load_dotenv()


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
        print(f"Fetching data for {coin_id}...")
        df = data_agent.fetch_data(coin_id, vs_currency, days)

        if df.empty:
            print(
                "Error: No data fetched. Please check your internet connection and API status.")
            return

        # Prepare data and train the model
        print("Preparing data and training the model...")
        X, y = prediction_engine.prepare_data(df)

        if X.empty or y.empty:
            print(
                "Error: No valid data after preparation. Please check the data quality and preparation process.")
            return

        prediction_engine.train(X, y)

        # Make a prediction using the machine learning model
        print("Making a prediction using the ML model...")
        latest_data = X.iloc[-1].to_frame().T
        ml_prediction, ml_probabilities = prediction_engine.predict(
            latest_data)

        # Prepare data for LLM prediction
        market_data = df.tail(
            3)[['price', 'high', 'low', 'volume']].to_string()
        technical_indicators = X.tail(3).to_string()

        # Make a prediction using the LLM
        print("Making a prediction using the LLM...")
        llm_prediction = llm_predictor.predict(
            market_data, technical_indicators)

        # Print results
        print("\n--- Market Summary ---")
        print(f"Current Price: ${df['price'].iloc[-1]:.2f}")
        print(f"24h High: ${df['high'].iloc[-1]:.2f}")
        print(f"24h Low: ${df['low'].iloc[-1]:.2f}")
        print(f"24h Price Range: ${df['high'].iloc[-1] - df['low'].iloc[-1]:.2f} ({
              ((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['low'].iloc[-1] * 100):.2f}%)")

        print("\n--- Machine Learning Model Prediction ---")
        print(f"Prediction: {'Bullish' if ml_prediction == 1 else 'Bearish'}")
        print(f"Confidence: {max(ml_probabilities):.2f}")

        print("\n--- LLM Model Prediction ---")
        print(llm_prediction)

        # Explain the ML model's prediction
        feature_importance = prediction_engine.explain_prediction(X)
        print("\n--- Top 5 Most Important Features (ML Model) ---")
        print(feature_importance.head())

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
