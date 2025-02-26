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

def get_erc20_balance_request(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Retrieve the balance of a specified ERC20 token for a given account address."""
    
    chain = kwargs.get("chain", "ethereum:mainnet")
    user = kwargs.get("user")
    token = kwargs.get("token")
    
    if not all([user, token]):
        return "Missing required parameters.", None, None, None
    
    url = "https://api.compasslabs.ai/v0/generic/balance/get"
    payload = {
        "chain": chain,
        "user": user,
        "token": token
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
    user = kwargs.get("user")
    token = kwargs.get("token")
    
    # Fetch balance
    error, balance_data, _, _ = get_erc20_balance_request(chain=chain, user=user, token=token)
    
    if error:
        return f"Failed to retrieve ERC20 balance: {error}", "", None, None
    
    # Generate prompt incorporating the balance data
    prompt = f"""
    You are a financial analyst providing insights into the ERC20 token balance for a user.
    Here is the retrieved balance data:
    {json.dumps(balance_data, indent=4)}
    Provide an analysis explaining the balance.
    """
    
    try:
        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing insights into ERC20 token balances."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        
        # Create response dictionary with all data
        metadata_dict = {
            "question": f"ERC20 token balance for {user}",
            "balance_data": balance_data,
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
            "question": f"ERC20 token balance for {user}",
            "timestamp": datetime.datetime.now().isoformat()
        }
        return f"Failed to generate analysis: {str(e)}", "", error_dict, None