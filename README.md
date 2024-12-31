# Crypto Prediction Agent

This project implements a Python-based crypto prediction agent that analyzes and predicts whether a given cryptocurrency will be bullish or bearish within a specified timeframe.

## Features

- Fetches cryptocurrency data from the CoinGecko API
- Calculates technical indicators using the `ta` library
- Performs sentiment analysis on cryptocurrency descriptions
- Uses a Random Forest Classifier to predict market trends
- Provides explanations for predictions based on feature importance

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/crypto-prediction-agent.git
   cd crypto-prediction-agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to fetch data, train the model, and make a prediction:

```
python crypto_prediction_agent/main.py
```

## Customization

- To analyze a different cryptocurrency, change the `coin_id` variable in `main.py`.
- To adjust the timeframe, modify the `days` variable in `main.py`.
- To use different technical indicators or features, update the `prepare_data` method in `prediction_engine.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.