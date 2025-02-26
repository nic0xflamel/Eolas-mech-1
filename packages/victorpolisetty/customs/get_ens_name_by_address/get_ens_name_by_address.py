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

def get_ens_address_request(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Retrieve the Ethereum wallet address associated with an ENS name."""
    
    chain = kwargs.get("chain", "ethereum:mainnet")
    ens_name = kwargs.get("ens_name")
    
    if not ens_name:
        return "Missing required parameters.", None, None, None
    
    url = "https://api.compasslabs.ai/v0/generic/ens/get"
    payload = {
        "chain": chain,
        "ens_name": ens_name
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
    ens_name = kwargs.get("ens_name")
    
    # Fetch ENS address
    error, ens_data, _, _ = get_ens_address_request(chain=chain, ens_name=ens_name)
    
    if error:
        return f"Failed to retrieve ENS address: {error}", "", None, None
    
    # Generate prompt incorporating the ENS data
    prompt = f"""
    Here is the retrieved ENS data:
    {json.dumps(ens_data)}
    Here is the ENS name given:
    {ens_name}
    """

    try:
        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Simply output the ENS name given and the address provided in a friendly way."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        
        # Create response dictionary with all data
        metadata_dict = {
            "question": f"ENS address lookup for {ens_name}",
            "ens_data": ens_data,
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
            "question": f"ENS address lookup for {ens_name}",
            "timestamp": datetime.datetime.now().isoformat()
        }
        return f"Failed to generate analysis: {str(e)}", "", error_dict, None