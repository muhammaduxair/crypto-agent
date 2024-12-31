from data_fetcher import DataFetchingAgent
from prediction_engine import PredictionEngine


def main():
    # Initialize agents
    data_agent = DataFetchingAgent()
    prediction_engine = PredictionEngine()

    # Set parameters
    coin_id = 'bitcoin'
    vs_currency = 'usd'
    days = 365

    # Fetch data
    print(f"Fetching data for {coin_id}...")
    df = data_agent.fetch_data(coin_id, vs_currency, days)

    # Prepare data and train the model
    print("Preparing data and training the model...")
    X, y = prediction_engine.prepare_data(df)
    prediction_engine.train(X, y)

    # Make a prediction for the next day
    print("Making a prediction for the next day...")
    latest_data = X.iloc[-1].to_frame().T
    prediction, probabilities = prediction_engine.predict(latest_data)

    # Explain the prediction
    feature_importance = prediction_engine.explain_prediction(X)

    # Print results
    print(f"\nPrediction for {coin_id} in the next 24 hours:")
    print(f"{'Bullish' if prediction == 1 else 'Bearish'} with {
          probabilities[prediction]:.2f} probability")

    print("\nTop 5 most important features:")
    print(feature_importance.head())


if __name__ == "__main__":
    main()
