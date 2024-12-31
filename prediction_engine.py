import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class PredictionEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def prepare_data(self, df):
        print(f"Input data shape: {df.shape}")
        print(f"Columns before preparation: {df.columns}")

        # Calculate daily returns
        df['returns'] = df['price'].pct_change()

        # Create target variable (1 for bullish, 0 for bearish)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        print(f"Data shape after creating target: {df.shape}")

        # Select features for the model
        features = ['volume', 'sentiment', 'high', 'low'] + \
            [col for col in df.columns if col.startswith(
                'momentum') or col.startswith('trend')]

        # Remove features with all NaN values
        features = [f for f in features if not df[f].isna().all()]

        X = df[features]
        y = df['target']

        # Forward fill NaN values
        X = X.fillna(method='ffill')

        # Backward fill any remaining NaN values at the beginning
        X = X.fillna(method='bfill')

        # Calculate additional features
        X['price_range'] = X['high'] - X['low']
        X['price_range_pct'] = (X['high'] - X['low']) / X['low']

        # Remove any rows that still have NaN values
        mask = X.isna().any(axis=1) | y.isna()
        X = X[~mask]
        y = y[~mask]

        print(f"Features: {features}")
        print(f"X shape after cleaning: {X.shape}")
        print(f"y shape after cleaning: {y.shape}")

        return X, y

    def train(self, X, y):
        if X.empty or y.empty:
            print("Error: Empty dataset. Unable to train the model.")
            return

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"Model accuracy: {accuracy:.2f}")

    def predict(self, X):
        if self.model is None:
            print(
                "Error: Model not trained. Please train the model before making predictions.")
            return None, None

        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return prediction[0], probabilities[0]

    def explain_prediction(self, X):
        if self.model is None:
            print(
                "Error: Model not trained. Please train the model before explaining predictions.")
            return None

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance

    def calculate_price_range(self, df, prediction, confidence, prediction_days):
        # Calculate daily returns
        df['daily_return'] = df['price'].pct_change()

        # Calculate historical volatility (standard deviation of returns)
        volatility = df['daily_return'].std()

        # Get the current price
        current_price = df['price'].iloc[-1]

        # Calculate the potential price range
        # Adjust volatility for the prediction timeframe
        adjusted_volatility = volatility * (prediction_days ** 0.5)
        lower_bound = current_price * (1 - adjusted_volatility)
        upper_bound = current_price * (1 + adjusted_volatility)

        # Adjust the range based on the prediction and confidence
        if prediction == 1:  # Bullish
            lower_bound += (upper_bound - lower_bound) * confidence * 0.5
        else:  # Bearish
            upper_bound -= (upper_bound - lower_bound) * confidence * 0.5

        return lower_bound, upper_bound
