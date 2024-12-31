import os
from groq import Groq
from typing import Dict, Optional
from datetime import datetime
import numpy as np


class LLMPredictor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM predictor with Groq API key.

        Args:
            api_key (str, optional): Groq API key. If not provided, will look for GROQ_API_KEY in environment variables.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please provide it or set GROQ_API_KEY environment variable.")

        self.client = Groq(api_key=self.api_key)

    def _validate_inputs(self, market_data: Dict, technical_indicators: Dict,
                         current_price: float, prediction_days: int) -> None:
        """Validate input parameters before making prediction."""
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValueError("Current price must be a positive number")

        if not isinstance(prediction_days, int) or prediction_days <= 0:
            raise ValueError("Prediction days must be a positive integer")

        if not market_data or not isinstance(market_data, dict):
            raise ValueError("Market data must be a non-empty dictionary")

        if not technical_indicators or not isinstance(technical_indicators, dict):
            raise ValueError(
                "Technical indicators must be a non-empty dictionary")

    def _format_market_data(self, market_data: Dict) -> str:
        """Format market data for the prompt."""
        formatted_data = []
        for key, value in market_data.items():
            if isinstance(value, (int, float)):
                formatted_data.append(f"{key}: {value:.2f}")
            else:
                formatted_data.append(f"{key}: {value}")
        return "\n".join(formatted_data)

    def _format_technical_indicators(self, indicators: Dict) -> str:
        """Format technical indicators for the prompt."""
        formatted_indicators = []
        for key, value in indicators.items():
            if isinstance(value, (int, float)):
                formatted_indicators.append(f"{key}: {value:.2f}")
            else:
                formatted_indicators.append(f"{key}: {value}")
        return "\n".join(formatted_indicators)

    def _parse_response(self, response_text: str) -> Dict:
        """Parse the LLM response into a structured format."""
        try:
            lines = response_text.strip().split('\n')
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction': None,
                'confidence': None,
                'price_range': {
                    'lower': None,
                    'upper': None
                },
                'reasoning': '',
                'key_factors': []
            }

            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('Prediction:'):
                    result['prediction'] = line.split(':')[1].strip()
                elif line.startswith('Confidence:'):
                    try:
                        result['confidence'] = float(
                            line.split(':')[1].strip())
                    except ValueError:
                        result['confidence'] = None
                elif line.startswith('- Lower Bound: $'):
                    try:
                        result['price_range']['lower'] = float(
                            line.split('$')[1].strip())
                    except ValueError:
                        result['price_range']['lower'] = None
                elif line.startswith('- Upper Bound: $'):
                    try:
                        result['price_range']['upper'] = float(
                            line.split('$')[1].strip())
                    except ValueError:
                        result['price_range']['upper'] = None
                elif line.startswith('Reasoning:'):
                    current_section = 'reasoning'
                elif line.startswith('Key Factors:'):
                    current_section = 'key_factors'
                elif line and current_section:
                    if current_section == 'reasoning':
                        result['reasoning'] += line + ' '
                    elif current_section == 'key_factors' and line.startswith('-'):
                        result['key_factors'].append(line[1:].strip())

            return result

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")

    def predict(self, market_data: Dict, technical_indicators: Dict,
                current_price: float, prediction_days: int,
                temperature: float = 0.2) -> Dict:
        """
        Make a market prediction using the Groq LLM.

        Args:
            market_data (Dict): Historical market data and metrics
            technical_indicators (Dict): Technical analysis indicators
            current_price (float): Current price of the asset
            prediction_days (int): Number of days to predict ahead
            temperature (float, optional): LLM temperature parameter. Defaults to 0.2.

        Returns:
            Dict: Structured prediction results including market direction, confidence, 
                 price range, reasoning, and key factors.
        """
        try:
            # Validate inputs
            self._validate_inputs(
                market_data, technical_indicators, current_price, prediction_days)

            # Format data for prompt
            formatted_market_data = self._format_market_data(market_data)
            formatted_indicators = self._format_technical_indicators(
                technical_indicators)

            # Create prompt
            prompt = f"""
            As a crypto market expert, analyze the following data and predict the market trend for the next {prediction_days} day{'s' if prediction_days > 1 else ''}. Provide a detailed analysis including a potential price range.

            Current Price: ${current_price:.2f}

            Market Data:
            {formatted_market_data}

            Technical Indicators:
            {formatted_indicators}

            Based on this information:
            1. Predict whether the market will be bullish or bearish in the next {prediction_days} day{'s' if prediction_days > 1 else ''}.
            2. Provide a confidence score between 0 and 1 for your prediction.
            3. Estimate a potential price range for the next {prediction_days} day{'s' if prediction_days > 1 else ''}, giving both a lower and upper bound.
            4. Explain your reasoning, considering factors such as recent price movements, volume trends, and key technical indicators.

            Format your response exactly as follows:
            Prediction: [Bullish/Bearish]
            Confidence: [0-1]
            Estimated Price Range:
            - Lower Bound: $X
            - Upper Bound: $Y
            Reasoning: [Your detailed analysis]
            Key Factors:
            - [Factor 1]
            - [Factor 2]
            - [Factor 3]
            """

            # Make API call
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert crypto analyst with deep knowledge of market trends and technical analysis. Always provide specific, numerical predictions and strictly follow the requested output format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="mixtral-8x7b-32768",
                temperature=temperature,
                max_tokens=1000,
            )

            # Parse and validate response
            prediction_result = self._parse_response(
                response.choices[0].message.content)

            # Add metadata
            prediction_result['metadata'] = {
                'model': "mixtral-8x7b-32768",
                'temperature': temperature,
                'prediction_days': prediction_days,
                'current_price': current_price
            }

            return prediction_result

        except Exception as e:
            error_result = {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
            return error_result

    def calculate_prediction_accuracy(self, historical_predictions: list) -> Dict:
        """
        Calculate the accuracy of past predictions.

        Args:
            historical_predictions (list): List of previous predictions and actual outcomes

        Returns:
            Dict: Accuracy metrics
        """
        if not historical_predictions:
            return {'error': 'No historical predictions provided'}

        try:
            total = len(historical_predictions)
            correct = sum(1 for pred in historical_predictions if pred.get(
                'was_correct', False))
            accuracy = correct / total if total > 0 else 0

            confidence_correlation = np.corrcoef(
                [pred['confidence'] for pred in historical_predictions],
                [pred['was_correct'] for pred in historical_predictions]
            )[0, 1]

            return {
                'total_predictions': total,
                'correct_predictions': correct,
                'accuracy': accuracy,
                'confidence_correlation': confidence_correlation
            }
        except Exception as e:
            return {'error': f'Failed to calculate accuracy: {str(e)}'}
