from datetime import datetime
import functools
from dotenv import load_dotenv
import openai
import requests
from typing import Any, Callable, Dict, Optional, Tuple
import json

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

def get_aave_position_summary_request(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Get Aave position summary or per-token summary if 'asset' is specified."""
    
    chain = kwargs.get("chain", "ethereum:mainnet")
    user = kwargs.get("user")
    asset = kwargs.get("asset")  # Optional parameter for per-token summary

    if not user:
        return "User address not provided.", None, None, None

    # Determine which endpoint and payload to use
    if asset:
        url = "https://api.compasslabs.ai/v0/aave/user_position_per_token/get"
        payload = {
            "chain": chain,
            "user": user,
            "asset": asset
        }
    else:
        url = "https://api.compasslabs.ai/v0/aave/user_position_summary/get"
        payload = {
            "chain": chain,
            "user": user
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
    # Get the question from prompt
    user = kwargs.get("user")
    chain = kwargs.get("chain", "ethereum:mainnet")    
    asset = kwargs.get("asset")  # Optional asset parameter

    # Generate prediction using OpenAI
    error, position_summary, _, _ = get_aave_position_summary_request(user=user, chain=chain, asset=asset)
    if error:
        return f"Failed to fetch Aave position summary: {error}", "", None, None
    
    # Generate prompt incorporating the position summary
    prompt = f"""
    You are a financial analyst explaining an Aave Position Summary to a slightly technical user.
    Here is the retrieved data:
    {json.dumps(position_summary, indent=4)}
    Provide an analysis explaining what this means in terms of collateral, debt, risk, and borrowing capacity.
    """
 
    try:
        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst giving a person who is slightly technical their Aave Position Summary."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        
        # Create response dictionary with all data
        metadata_dict = {
            "question": f"Aave position summary for user {user} {f'and asset {asset}' if asset else ''}",
            "position_summary": position_summary,
            "timestamp": datetime.now().isoformat()
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
            "question": f"Aave position summary for user {user} {f'and asset {asset}' if asset else ''}",
            "timestamp": datetime.now().isoformat()
        }
        return f"Failed to generate prediction: {str(e)}", "", error_dict, None