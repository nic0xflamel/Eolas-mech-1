import functools

import openai
import requests
from typing import Dict, Optional, Tuple, Any, Callable
from openai import OpenAI
from dotenv import load_dotenv
import os

import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# Load environment variables
load_dotenv()


class APIClients:
    def __init__(self):
        self.synth_api_key = os.getenv("SYNTH_API_KEY")
        
        if not all([self.synth_api_key]):
            raise ValueError("Missing required API keys in environment variables")
        

def get_btc_predictions(clients: APIClients) -> Optional[list]:
    """Fetches BTC price predictions from the Synth API.
    
    Makes a GET request to the Synth API endpoint to retrieve price predictions for Bitcoin.
    The predictions are for the next 24 hours with 10-minute intervals.
    
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
        "time_increment": 300,  # 10 minutes in seconds
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
    
    print("Successfully fetched BTC price predictions")
    
    return data


def get_most_likely_prediction(data: list) -> list:
    """Find the most likely price prediction by analyzing all simulations.
    
    Args:
        data: List of prediction data containing simulations
        
    Returns:
        A list of 24 hourly price points representing the most likely price trajectory,
        where each point contains the time and the most frequently predicted price level.
    """
    # Dictionary to store all prices for each timestamp
    time_to_prices = {}
    
    for prediction in data:
        simulations = prediction["prediction"]
        
        for simulation in simulations:
            for point in simulation:
                time = point["time"]
                price = point["price"]
                
                if time not in time_to_prices:
                    time_to_prices[time] = []
                time_to_prices[time].append(price)
    
    # Get most likely price for each timestamp
    most_likely_prediction = []
    sorted_times = sorted(time_to_prices.keys())
    
    for time in sorted_times:
        prices = time_to_prices[time]
        # Use median price as most likely price level
        most_likely_price = sorted(prices)[len(prices)//2]
        
        most_likely_prediction.append({
            "time": time,
            "price": most_likely_price
        })

    print("Generated most likely prediction")
    
    return most_likely_prediction


def get_price_prediction_chart(best_prediction_data: list, btc_predictions: list) -> str:
        """Generate a price prediction chart showing the median prediction and individual simulations.
        
        Args:
            best_prediction_data: List of dictionaries containing the median price prediction data points
            btc_predictions: List of dictionaries containing all simulation data
            
        Returns:
            Base64 encoded chart image with data URI prefix for direct PNG viewing
        """
        print("Generating price prediction chart")

        # Extract data from dictionaries into lists
        datetimes = [pd.to_datetime(item['time']).timestamp() for item in best_prediction_data]
        prices = [item['price'] for item in best_prediction_data]
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': datetimes,
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': prices
        })
        # Convert datetime strings to datetime objects
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Create figure
        fig, price_ax = plt.subplots(figsize=(12, 8))
        price_ax.plot(df.close, label='Median Price Prediction', color='blue')
        price_ax.set_title('BTC Price Prediction')
        price_ax.set_ylabel('Price')
        price_ax.grid(True)

        # Set only the left x-axis limit to match data start
        left_limit = df.index[0]
        price_ax.set_xlim(left=left_limit)

        # Plot all prediction simulations
        for prediction in btc_predictions[0]["prediction"]:
            datetimes = [pd.to_datetime(item['time']).timestamp() for item in prediction]
            prices = [item['price'] for item in prediction]
            
            # Convert timestamps to datetime and create a Series
            dates = pd.to_datetime(datetimes, unit='s')
            prediction_series = pd.Series(prices, index=dates)
            
            # Plot with lower alpha for visibility
            price_ax.plot(prediction_series, alpha=0.2, color='grey', label='_nolegend_')

        # Add legend to price plot
        price_ax.legend(loc='upper right')
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save chart to bytes
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Encode as base64 string with data URI prefix for PNG
        chart_b64 = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        
        print("Chart generation completed successfully")
        return chart_b64


def run(
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run Bitcoin price prediction analysis and generate price prediction chart.
    
    Fetches Bitcoin price predictions from Synth API, processes the data to find the most likely
    price trajectory, and generates a visualization showing the median prediction alongside
    individual simulation paths.
    
    Args:
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tuple containing:
        - str: Base64 encoded PNG chart image with data URI prefix, or error message if failed
        - Optional[str]: Empty string (unused)
        - Optional[Dict[str, Any]]: List of prediction data points if successful, None if failed
        - Any: None (unused)
    """
    try:
        clients = APIClients()

        btc_predictions = get_btc_predictions(clients=clients)

        if not btc_predictions:
            return f"Failed to get BTC predictions from Synth subnet", "", None, None

        most_likely_prediction = get_most_likely_prediction(btc_predictions)

        chart = get_price_prediction_chart(most_likely_prediction, btc_predictions)

        return chart, "", most_likely_prediction, None

    except Exception as e:
        print(f"Error in btc price prediction: {str(e)}")
        return str(e), "", None, None
