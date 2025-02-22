import requests
from typing import Any, Dict, Optional, Tuple

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
