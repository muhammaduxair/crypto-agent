import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.exceptions import NotFittedError
import logging
from typing import Tuple, Optional, Dict, Any


class PredictionEngine:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the prediction engine with customizable parameters.

        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced',
            max_features='sqrt',
            n_jobs=-1
        )
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.imputer = SimpleImputer(
            strategy='median')  # More robust than mean
        self.feature_columns = None
        self.is_fitted = False

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional technical indicators."""
        try:
            df = df.copy()

            # Price-based features
            df['log_return'] = np.log(df['price']).diff()
            df['volatility'] = df['log_return'].rolling(window=20).std()
            df['price_momentum'] = df['price'].pct_change(periods=5)
            df['price_acceleration'] = df['price_momentum'].diff()

            # Volume-based features
            df['volume_momentum'] = df['volume'].pct_change(periods=5)
            df['volume_ma_ratio'] = df['volume'] / \
                df['volume'].rolling(window=20).mean()

            # Price range features
            df['daily_range'] = df['high'] - df['low']
            df['range_ratio'] = df['daily_range'] / df['price']
            df['range_ma_ratio'] = df['daily_range'] / \
                df['daily_range'].rolling(window=20).mean()

            # Bollinger Bands
            df['bb_middle'] = df['price'].rolling(window=20).mean()
            df['bb_std'] = df['price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['price'] - df['bb_lower']) / \
                (df['bb_upper'] - df['bb_lower'])

            # RSI-like features
            gains = df['log_return'].copy()
            losses = df['log_return'].copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            df['avg_gain'] = gains.rolling(window=14).mean()
            df['avg_loss'] = abs(losses.rolling(window=14).mean())
            df['gain_loss_ratio'] = df['avg_gain'] / df['avg_loss']

            return df

        except Exception as e:
            self.logger.error(f"Error in feature creation: {str(e)}")
            raise

    def prepare_data(self, df: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training or prediction.

        Args:
            df (pd.DataFrame): Input dataframe
            target_horizon (int): Number of periods ahead to predict

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Prepared features and target
        """
        try:
            self.logger.info(
                f"Starting data preparation. Input shape: {df.shape}")

            # Create technical features
            df = self._create_technical_features(df)

            # Create target variable
            df['target'] = (df['price'].shift(-target_horizon)
                            > df['price']).astype(int)

            # Select features
            base_features = [
                'volume', 'sentiment', 'high', 'low', 'log_return', 'volatility',
                'price_momentum', 'price_acceleration', 'volume_momentum', 'volume_ma_ratio',
                'daily_range', 'range_ratio', 'range_ma_ratio', 'bb_position',
                'gain_loss_ratio'
            ]

            # Add any momentum or trend features from the original dataset
            additional_features = [
                col for col in df.columns if col.startswith(('momentum_', 'trend_'))]
            features = base_features + additional_features

            # Remove features with too many NaN values
            valid_features = [
                f for f in features if f in df.columns and df[f].isna().mean() < 0.1]

            X = df[valid_features]
            y = df['target']

            # Handle missing values
            X = pd.DataFrame(self.imputer.fit_transform(
                X), columns=valid_features, index=X.index)

            # Store feature columns for later use
            self.feature_columns = valid_features

            # Remove rows with NaN targets
            mask = y.notna()
            X = X.loc[mask]
            y = y.loc[mask]

            self.logger.info(
                f"Data preparation complete. Output shapes - X: {X.shape}, y: {y.shape}")
            return X, y

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the model and evaluate its performance.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable

        Returns:
            Dict[str, Any]: Dictionary containing model performance metrics
        """
        try:
            if X.empty or y.empty:
                raise ValueError("Empty dataset provided for training")

            # Create time series cross-validation splits
            tscv = TimeSeriesSplit(n_splits=5)

            # Scale the features
            X_scaled = self.scaler.fit_transform(X)

            # Perform cross-validation
            cv_scores = cross_val_score(
                self.model, X_scaled, y, cv=tscv, scoring='roc_auc')

            # Train the final model
            self.model.fit(X_scaled, y)
            self.is_fitted = True

            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Calculate training metrics
            y_pred = self.model.predict(X_scaled)
            y_prob = self.model.predict_proba(X_scaled)

            metrics = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'roc_auc': roc_auc_score(y, y_prob[:, 1]),
                'classification_report': classification_report(y, y_pred),
                'feature_importance': feature_importance,
                'confusion_matrix': confusion_matrix(y, y_pred)
            }

            self.logger.info(
                f"Model training complete. CV ROC-AUC: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> Tuple[int, float, Dict[str, float]]:
        """
        Make predictions with confidence scores and prediction intervals.

        Args:
            X (pd.DataFrame): Feature matrix

        Returns:
            Tuple[int, float, Dict[str, float]]: Prediction, confidence, and additional metrics
        """
        try:
            if not self.is_fitted:
                raise NotFittedError("Model has not been trained yet")

            if not all(col in X.columns for col in self.feature_columns):
                raise ValueError(
                    "Input features don't match training features")

            X = X[self.feature_columns]
            X = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X)

            # Get predictions and probabilities
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)

            # Get individual tree predictions for uncertainty estimation
            tree_predictions = np.array(
                [tree.predict(X_scaled) for tree in self.model.estimators_])
            prediction_std = tree_predictions.std()

            metrics = {
                'confidence': confidence,
                'probability_bullish': probabilities[1],
                'probability_bearish': probabilities[0],
                'prediction_std': prediction_std
            }

            return prediction, confidence, metrics

        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

    def calculate_price_range(self, df: pd.DataFrame, prediction: int, confidence: float,
                              prediction_days: int) -> Tuple[float, float]:
        """
        Calculate expected price range based on prediction and historical volatility.

        Args:
            df (pd.DataFrame): Historical price data
            prediction (int): Model prediction (0 or 1)
            confidence (float): Prediction confidence
            prediction_days (int): Forecast horizon

        Returns:
            Tuple[float, float]: Lower and upper price bounds
        """
        try:
            # Calculate log returns and volatility
            df['log_return'] = np.log(df['price']).diff()
            volatility = df['log_return'].std()
            current_price = df['price'].iloc[-1]

            # Adjust volatility for prediction timeframe
            adjusted_volatility = volatility * np.sqrt(prediction_days)

            # Calculate basic range
            z_score = 1.96  # 95% confidence interval
            basic_range = adjusted_volatility * z_score

            # Adjust range based on prediction and confidence
            if prediction == 1:  # Bullish
                upper_bound = current_price * \
                    np.exp(basic_range * (1 + confidence * 0.5))
                lower_bound = current_price * \
                    np.exp(-basic_range * (1 - confidence * 0.5))
            else:  # Bearish
                upper_bound = current_price * \
                    np.exp(basic_range * (1 - confidence * 0.5))
                lower_bound = current_price * \
                    np.exp(-basic_range * (1 + confidence * 0.5))

            return lower_bound, upper_bound

        except Exception as e:
            self.logger.error(f"Error in price range calculation: {str(e)}")
            raise

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and performance metrics.

        Returns:
            Dict[str, Any]: Dictionary containing model diagnostics
        """
        if not self.is_fitted:
            return {"error": "Model not fitted yet"}

        return {
            "feature_importance": pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False),
            "n_features": len(self.feature_columns),
            "model_params": self.model.get_params(),
            "n_classes": len(self.model.classes_),
            "n_estimators": len(self.model.estimators_)
        }
