import functools
import json
import datetime
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import openai
import requests

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Load environment variables
load_dotenv()

class APIClients:
    def __init__(self, api_keys: Any):
        self.openai_api_key = api_keys["openai"]

        if not all([self.openai_api_key]):
            raise ValueError("Missing required API keys")

        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

def with_key_rotation(func: Any):
    """Decorator to handle API key rotation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]

        def execute() -> MechResponse:
            try:
                return func(*args, **kwargs)  # Ensure it returns exactly 4 values
            except Exception as e:
                return str(e), "", None, None  # Ensure it returns exactly 4 values

        return execute()

    return wrapper

def get_aave_asset_price_request(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Get Aave asset price from the Aave V3 Oracle."""
    
    chain = kwargs.get("chain", "ethereum:mainnet")
    asset = kwargs.get("asset")
    
    if not asset:
        return "Asset identifier not provided.", None, None, None
    
    url = "https://api.compasslabs.ai/v0/aave/asset_price/get"
    payload = {
        "chain": chain,
        "asset": asset
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return None, data, None, None
        else:
            return f"Error: {response.status_code} {response.text}", None, None, None
    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None

@with_key_rotation
def run(**kwargs) -> MechResponse:
    """Run the task"""
    # Initialize API clients
    clients = APIClients(kwargs["api_keys"])
    
    # Get parameters
    chain = kwargs.get("chain", "ethereum:mainnet")    
    asset = kwargs.get("asset")
    
    # Fetch asset price
    error, asset_price, _, _ = get_aave_asset_price_request(chain=chain, asset=asset)
    
    if error:
        return f"Failed to fetch Aave asset price: {error}", "", None, None
    
    # Generate prompt incorporating the asset price
    prompt = f"""
    You are a financial analyst giving an asset price. The price is recieved using the Aave API.
    Here is the correct price for the asset {asset}:
    {json.dumps(asset_price, indent=4)}
    """
    
    try:
        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing an asset price from the Aave API."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        
        # Create response dictionary with all data
        metadata_dict = {
            "question": f"Asset price for {asset}",
            "asset_price": asset_price,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return (
            response_text,
            "",
            metadata_dict,
            None,
        ) 
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "question": f"Asset price for {asset}",
            "timestamp": datetime.datetime.now().isoformat()
        }
        return f"Failed to generate analysis: {str(e)}", "", error_dict, None

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Define test inputs
    test_kwargs = {
        "api_keys": {
            "openai": os.getenv("OPENAI_API_KEY")
        },
        "chain": "ethereum:mainnet",
        "asset": "WETH"
    }

    # Run the function
    response_text, _, metadata, _ = run(**test_kwargs)

    # Print results
    print("\n=== Response from GPT-4o ===\n")
    print(response_text)

    print("\n=== Metadata ===\n")
    print(metadata)
