import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class PredictionEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        # Calculate daily returns
        df['returns'] = df['price'].pct_change()

        # Create target variable (1 for bullish, 0 for bearish)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Select features for the model
        features = ['volume', 'sentiment'] + \
            [col for col in df.columns if col.startswith(
                'momentum') or col.startswith('trend')]
        X = df[features]
        y = df['target']

        return X, y

    def train(self, X, y):
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
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return prediction[0], probabilities[0]

    def explain_prediction(self, X):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance
