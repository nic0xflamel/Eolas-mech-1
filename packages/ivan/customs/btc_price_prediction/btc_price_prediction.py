import functools

import openai
import requests
from typing import Dict, Optional, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime


class APIClients:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        self.synth_api_key = os.getenv("SYNTH_API_KEY")
        self.taapi_api_key = os.getenv("TAAPI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([self.synth_api_key, self.taapi_api_key, self.openai_api_key]):
            raise ValueError("Missing required API keys in environment variables")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        

def get_btc_predictions(clients: APIClients) -> Optional[list]:
    """Fetches BTC price predictions from the Synth API.
    
    Makes a GET request to the Synth API endpoint to retrieve price predictions for Bitcoin.
    The predictions are for the next 24 hours with 5-minute intervals.
    
    Args:
        clients: APIClients instance containing the required Synth API key
        
    Returns:
        Optional[list]: List of prediction data if successful, None if the API request fails
        or no predictions are available
    """
    synth_api_key = clients.synth_api_key
    endpoint = "https://synth.mode.network/prediction/best"
    
    # Set up parameters
    params = {
        "asset": "BTC",
        "time_increment": 300,  # 5 minutes in seconds
        "time_length": 86400    # 24 hours in seconds
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Apikey {synth_api_key}"
    }
    
    response = requests.get(endpoint, params=params, headers=headers)
    if response.status_code != 200:
        print(f"API request failed with status code: {response.status_code}")
        print(f"Response content: {response.text}")  # This will show the error message from the API
        return None
    
    data = response.json()
    if not data:  # This will catch both None and empty list/dict responses
        print(f"No predictions available for parameters: {params}")
        return None
    
    return data


def get_price_after_24hrs(data: list) -> float:
    """Get the median predicted price from the final data point of all simulations.
    
    Args:
        data: List of prediction data containing simulations
        
    Returns:
        float: The median predicted price from the final data points of all simulations
    """
    final_prices = []

    print("Calculating median price after 24hrs")
    
    for prediction in data:
        simulations = prediction["prediction"]
        
        for simulation in simulations:
            # Get the last price point from each simulation
            final_prices.append(simulation[-1]["price"])
    
    # Calculate median of final prices
    median_price = sorted(final_prices)[len(final_prices)//2]
    
    print(f"Generated price prediction 24hrs from now: ${median_price:,.2f}")
    
    return median_price


def get_btc_ta_data(clients: APIClients) -> Optional[dict]:
    """Get BTC price data from TAAPI.
    
    Args:
        clients: APIClients instance containing the TAAPI API key
        
    Returns:
        Optional[dict]: BTC price data if successful, None if the API request fails
    """
    taapi_api_key = clients.taapi_api_key

    print("Getting TAAPI data")

    def get_indicator_data(indicator: str, interval: str) -> Optional[dict]:
        endpoint = f"https://api.taapi.io/{indicator}"
  
        # Define a parameters dict for the parameters to be sent to the API 
        parameters = {
            'secret': taapi_api_key,
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'interval': interval
        } 
  
        # Send get request and save the response as response object 
        response = requests.get(url = endpoint, params = parameters)
  
        # Extract data in json format 
        return response.json() 
    
    pivot_points_data = get_indicator_data("pivotpoints", "1d")

    fibonacci_data = get_indicator_data("fibonacciretracement", "1h")

    return {
        "pivot_points": pivot_points_data,
        "fibonacci": fibonacci_data
    }


def get_prediction_report(clients: APIClients, price_after_24hrs: float, btc_ta_data: dict) -> str:
    """Generate a detailed Bitcoin price prediction report using technical analysis data.
    
    Formats pivot points and Fibonacci retracement data, then uses the DeepSeek API to analyze
    the technical indicators and generate a comprehensive price prediction report for the next 24 hours.
    
    Args:
        clients: APIClients instance containing the DeepSeek API client
        price_after_24hrs: Predicted Bitcoin price after 24 hours
        btc_ta_data: Dictionary containing technical analysis data from TAAPI including:
            - pivot_points: Dict with r3, r2, r1, p, s1, s2, s3 levels
            - fibonacci: Dict with retracement value, trend, price range and timestamps
            
    Returns:
        str: A detailed analysis report of likely BTC price movements over next 24 hours,
             including pivot point analysis, Fibonacci trends, and key price levels
    """

    print("Fetching prediction report")

    formatted_pivot_points = {
        "r3": btc_ta_data["pivot_points"]["r3"],
        "r2": btc_ta_data["pivot_points"]["r2"],
        "r1": btc_ta_data["pivot_points"]["r1"],
        "p": btc_ta_data["pivot_points"]["p"],
        "s1": btc_ta_data["pivot_points"]["s1"],
        "s2": btc_ta_data["pivot_points"]["s2"],
        "s3": btc_ta_data["pivot_points"]["s3"]
    }

    formatted_fibonacci = {
        "value": btc_ta_data["fibonacci"]["value"],
        "trend": btc_ta_data["fibonacci"]["trend"],
        "startPrice": btc_ta_data["fibonacci"]["startPrice"],
        "endPrice": btc_ta_data["fibonacci"]["endPrice"],
        "startTimestamp": datetime.fromtimestamp(btc_ta_data["fibonacci"]["startTimestamp"] / 1000).strftime("%d/%m/%Y %H:%M:%S"),
        "endTimestamp": datetime.fromtimestamp(btc_ta_data["fibonacci"]["endTimestamp"] / 1000).strftime("%d/%m/%Y %H:%M:%S")
    }

    prompt = f"""Analyze Bitcoin's price movement over the next 24 hours based on the following data:

1. Median Price Prediction in 24hrs: ${price_after_24hrs:,.2f}

2. Current Pivot Points:
- Resistance 3 (R3): ${formatted_pivot_points['r3']:,.2f}
- Resistance 2 (R2): ${formatted_pivot_points['r2']:,.2f}
- Resistance 1 (R1): ${formatted_pivot_points['r1']:,.2f}
- Pivot Point (P): ${formatted_pivot_points['p']:,.2f}
- Support 1 (S1): ${formatted_pivot_points['s1']:,.2f}
- Support 2 (S2): ${formatted_pivot_points['s2']:,.2f}
- Support 3 (S3): ${formatted_pivot_points['s3']:,.2f}

3. Fibonacci Retracement:
- Current Value: {formatted_fibonacci['value']}
- Trend: {formatted_fibonacci['trend']}
- Price Range: ${formatted_fibonacci['startPrice']:,.2f} to ${formatted_fibonacci['endPrice']:,.2f}
- Time Range: {formatted_fibonacci['startTimestamp']} to {formatted_fibonacci['endTimestamp']}

Please provide a detailed analysis of the likely price movement over the next 24 hours. Consider:
- How the predicted price compares to current pivot points
- What the Fibonacci retracement suggests about trend strength
- Key support and resistance levels to watch
- Overall market sentiment and potential price targets

Do not mention that the data was provided to you by the user. Simply give the prediction report and do not provide any other contextualization."""

    response = clients.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cryptocurrency market analyst specializing in technical analysis and price predictions."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def run(
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run Bitcoin price prediction analysis using technical indicators and AI analysis.
    
    Fetches Bitcoin technical analysis data from TAAPI API and price predictions from Synth API.
    Combines this data to generate a detailed prediction report using DeepSeek AI that analyzes
    pivot points, Fibonacci retracements, and predicted price movements over the next 24 hours.
    
    Args:
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tuple containing:
        - str: Detailed prediction report text, or error message if failed
        - Optional[str]: Empty string (unused) 
        - Optional[Dict[str, Any]]: None (unused)
        - Any: None (unused)
    """
    try:
        clients = APIClients()

        btc_predictions = get_btc_predictions(clients=clients)

        if not btc_predictions:
            return f"Failed to get BTC simulation predictions from Synth subnet", "", None, None

        price_after_24hrs = get_price_after_24hrs(btc_predictions)

        btc_ta_data = get_btc_ta_data(clients=clients)

        if not btc_ta_data:
            return f"Failed to get BTC TA data from TAAPI", "", None, None

        prediction_report = get_prediction_report(clients=clients, price_after_24hrs=price_after_24hrs, btc_ta_data=btc_ta_data)

        if not prediction_report:
            return f"Failed to get prediction report from OpenAI", "", None, None

        return prediction_report, "", None, None

    except Exception as e:
        print(f"Error in btc price prediction: {str(e)}")
        return str(e), "", None, None