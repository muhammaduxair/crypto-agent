import os
from groq import Groq


class LLMPredictor:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def predict(self, market_data, technical_indicators, current_price, prediction_days):
        prompt = f"""
        As a crypto market expert, analyze the following data and predict the market trend for the next {prediction_days} day{'s' if prediction_days > 1 else ''}. Provide a detailed analysis including a potential price range.

        Current Price: ${current_price:.2f}

        Market Data:
        {market_data}

        Technical Indicators:
        {technical_indicators}

        Based on this information:
        1. Predict whether the market will be bullish or bearish in the next {prediction_days} day{'s' if prediction_days > 1 else ''}.
        2. Provide a confidence score between 0 and 1 for your prediction.
        3. Estimate a potential price range for the next {prediction_days} day{'s' if prediction_days > 1 else ''}, giving both a lower and upper bound.
        4. Explain your reasoning, considering factors such as recent price movements, volume trends, and key technical indicators.

        Format your response as:
        Prediction: [Bullish/Bearish]
        Confidence: [0-1]
        Estimated Price Range:
        - Lower Bound: $X
        - Upper Bound: $Y
        Reasoning: [Your detailed analysis]
        Key Factors: [List the top 3-5 factors influencing your prediction]
        """

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert crypto analyst with deep knowledge of market trends and technical analysis.",
                },
                {"role": "user", "content": prompt},
            ],
            model="mixtral-8x7b-32768",
            temperature=0.2,
            max_tokens=1000,
        )

        return response.choices[0].message.content
