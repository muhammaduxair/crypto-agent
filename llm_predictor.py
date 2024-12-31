import os
from groq import Groq


class LLMPredictor:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def predict(self, market_data, technical_indicators):
        prompt = f"""
        As a crypto market expert, analyze the following data and predict whether the market will be bullish or bearish in the next 24 hours. Provide a confidence score between 0 and 1.

        Market Data:
        {market_data}

        Technical Indicators:
        {technical_indicators}

        Format your response as:
        Prediction: [Bullish/Bearish]
        Confidence: [0-1]
        Reasoning: [Your analysis]
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
            max_tokens=500,
        )

        return response.choices[0].message.content
