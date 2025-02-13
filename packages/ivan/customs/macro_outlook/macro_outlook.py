import functools

import openai
import requests
from typing import Dict, Optional, Tuple, Any, Callable
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class APIClients:
    def __init__(self):
        self.synth_api_key = os.getenv("SYNTH_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        
        if not all([self.synth_api_key, self.perplexity_api_key]):
            raise ValueError("Missing required API keys in environment variables")
            
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )


def get_btc_predictions(clients: APIClients) -> Optional[list]:
    synth_api_key = clients.synth_api_key
    endpoint = "https://synth.mode.network/prediction/best"
    
    # Set up parameters
    params = {
        "asset": "BTC",
        "time_increment": 300,  # 5 minutes in seconds
        "time_length": 86400   # 24 hours in seconds
    }
    
    headers = {
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


def get_relevant_data(data: list) -> list:
    """Extract hourly price predictions from each simulation.
    
    Args:
        data: List of prediction data containing simulations
        
    Returns:
        List of simulations, where each simulation is a list of hourly price points.
        Only every other simulation is included.
    """
    relevant_data = []
    
    for prediction in data:
        simulations = prediction["prediction"]
        
        for i, simulation in enumerate(simulations):
            # Skip every other simulation
            if i % 2 != 0:
                continue
                
            hourly_points = []
            first_point = simulation[0]
            current_time = first_point["time"]
            hourly_points.append(first_point)
            
            # Get one point per hour
            for point in simulation[1:]:
                point_time = point["time"]
                # Check if this point is 1 hour after the last recorded point
                if point_time > current_time and "T" in point_time:
                    time_diff = int(point_time[11:13]) - int(current_time[11:13])
                    if time_diff == 1 or time_diff == -23:  # Handle day rollover
                        hourly_points.append(point)
                        current_time = point_time
                        
                        # Break if we have 24 hourly points
                        if len(hourly_points) == 24:
                            break
            
            if len(hourly_points) == 24:
                relevant_data.append(hourly_points)
                
    return relevant_data


def get_macro_outlook(clients: APIClients, btc_predictions: list) -> Optional[str]:
    """Generate a macro outlook analysis for Bitcoin using simulated price predictions.
    
    Args:
        clients: APIClients instance containing OpenAI client
        btc_predictions: List of simulated price predictions, where each simulation contains
                        24 hourly price points
        
    Returns:
        A string containing the macro outlook analysis, or None if analysis fails
    """
    if not btc_predictions:
        return None
        
    try:
        # Format the predictions data for the prompt
        simulations_text = []
        start_time = btc_predictions[0][0]["time"] if btc_predictions else None
        
        for i, simulation in enumerate(btc_predictions):
            sim_text = "\n"
            for point in simulation:
                sim_text += f"\n- Time: {point['time']}, Price: ${point['price']:,.2f}"
            simulations_text.append(sim_text)
            
        prompt = f"""Based on prediction data, please provide a comprehensive macro outlook report for Bitcoin price movements over the next 24 hours, starting from {start_time}.

The prediction data represents different possible price trajectories:
{''.join(simulations_text)}

Please create a detailed macro outlook report that includes:
1. Overall price trend analysis based on the prediction data
2. Key price levels and potential support/resistance zones
3. Volatility assessment
4. Risk factors and potential price movement catalysts
5. Summary and conclusion

Focus on identifying broad patterns and insights across the prediction data. Do not reference individual predictions or simulations in your analysis. Present findings as general trends and observations."""

        response = clients.perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online", 
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating macro outlook analysis: {e}")
        return None


def run(
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run Bitcoin price prediction analysis and generate macro outlook report.
    
    Fetches Bitcoin price predictions, processes the data, and generates a comprehensive 
    macro outlook analysis report using AI.
    
    Args:
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tuple containing:
        - str: The macro outlook report or error message
        - Optional[str]: Empty string (unused)
        - Optional[Dict[str, Any]]: None (unused) 
        - Any: None (unused)
    """
    try:
        clients = APIClients()

        btc_predictions = get_btc_predictions(clients=clients)

        if not btc_predictions:
            return f"Failed to get BTC predictions from Synth subnet", "", None, None

        relevant_data = get_relevant_data(btc_predictions)

        macro_outlook = get_macro_outlook(clients=clients, btc_predictions=relevant_data)
        if not macro_outlook:
           return f"Failed to generate macro outlook for BTC", "", None, None

        return macro_outlook, "", None, None

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return str(e), "", None, None
