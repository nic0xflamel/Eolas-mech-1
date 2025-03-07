from typing import Any, Dict, Optional, Tuple
import compass.api_client
from compass.api_client.rest import ApiException
from pprint import pprint

class CompassAPIClient:
    """Singleton class to manage Compass API client instance."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            configuration = compass.api_client.Configuration(
                host="https://api.compasslabs.ai"
            )
            cls._instance = compass.api_client.ApiClient(configuration)
        return cls._instance
# Get the shared API client instance
api_client = CompassAPIClient()
# Supply collateral to earn interest or borrow against
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "sender": "0x8A2c9eD8F6B9aD09036Cc0F5AAcaE7E6708f3D0c",
#   "call_data": {
#     "asset": "1INCH",
#     "amount": 0,
#     "on_behalf_of": "string"
#   }
# }
def post_aave_supply(json_data: str = "{}") -> Dict[str, Any]:
    """ Processes an Aave supply transaction using JSON input. """
    from compass.api_client.models.base_transaction_request_aave_supply_call_data import BaseTransactionRequestAaveSupplyCallData
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        supply_request_instance = BaseTransactionRequestAaveSupplyCallData.from_json(json_data)

        # 🚀 **Make the actual API request to process the supply transaction**
        api_response = api_instance.process_request_v0_aave_supply_post(supply_request_instance)

        # Convert response to dictionary
        return {"supply_transaction_data": api_response.to_dict()}

    except Exception as e:
        return {"success": False, "error": str(e)}
# Borrow against your collateral
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "sender": "0x8A2c9eD8F6B9aD09036Cc0F5AAcaE7E6708f3D0c",
#   "call_data": {
#     "asset": "1INCH",
#     "amount": 0,
#     "interest_rate_mode": 1,
#     "on_behalf_of": "string"
#   }
# }
def post_aave_borrow(json_data: str = "{}") -> Dict[str, Any]:
    """ Processes an Aave borrow transaction using JSON input. """
    from compass.api_client.models.base_transaction_request_aave_borrow_call_data import BaseTransactionRequestAaveBorrowCallData
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        borrow_request_instance = BaseTransactionRequestAaveBorrowCallData.from_json(json_data)

        # 🚀 **Make the actual API request to process the borrow transaction**
        api_response = api_instance.process_request_v0_aave_borrow_post(borrow_request_instance)

        # Convert response to dictionary
        return {"borrow_transaction_data": api_response.to_dict()}

    except Exception as e:
        return {"success": False, "error": str(e)}
# Repay some or all tokens you borrowed
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "sender": "0x8A2c9eD8F6B9aD09036Cc0F5AAcaE7E6708f3D0c",
#   "call_data": {
#     "asset": "1INCH",
#     "amount": 0,
#     "interest_rate_mode": 1,
#     "on_behalf_of": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
#   }
# }
def post_aave_repay_borrowed(json_data: str = "{}") -> Dict[str, Any]:
    """ Processes an Aave repayment transaction using JSON input. """
    from compass.api_client.models.base_transaction_request_aave_repay_call_data import BaseTransactionRequestAaveRepayCallData
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        repay_request_instance = BaseTransactionRequestAaveRepayCallData.from_json(json_data)

        # 🚀 **Make the actual API request to process the repayment**
        api_response = api_instance.process_request_v0_aave_repay_post(repay_request_instance)

        # Convert response to dictionary
        return {"repay_transaction_data": api_response.to_dict()}

    except Exception as e:
        return {"success": False, "error": str(e)}
# Withdraw some or all of your collateral
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "sender": "0x8A2c9eD8F6B9aD09036Cc0F5AAcaE7E6708f3D0c",
#   "call_data": {
#     "asset": "1INCH",
#     "amount": 0,
#     "recipient": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
#   }
# }
def post_aave_withdraw_collateral(json_data: str = "{}") -> Dict[str, Any]:
    """ Processes an Aave collateral withdrawal transaction using JSON input. """
    from compass.api_client.models.base_transaction_request_aave_withdraw_call_data import BaseTransactionRequestAaveWithdrawCallData
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        withdraw_request_instance = BaseTransactionRequestAaveWithdrawCallData.from_json(json_data)

        # 🚀 **Make the actual API request to process the collateral withdrawal**
        api_response = api_instance.process_request_v0_aave_withdraw_post(withdraw_request_instance)

        # Convert response to dictionary
        return {"withdraw_transaction_data": api_response.to_dict()}

    except Exception as e:
        return {"success": False, "error": str(e)}
# Get the price of an asset in USD according to Aave
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "asset": "1INCH"
# }
def get_aave_asset_price(json_data: str = "{}") -> Dict[str, Any]:
    """ Fetches the price of an asset in USD from Aave using JSON input. """
    from compass.api_client.models.aave_get_asset_price import AaveGetAssetPrice
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        aave_get_asset_price_instance = AaveGetAssetPrice.from_json(json_data)

        # Make the actual API request
        api_response = api_instance.process_request_v0_aave_asset_price_get_post(aave_get_asset_price_instance)

        # Convert response to dictionary
        return {"price_data": api_response.to_dict()}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
# Get a summary of the user's position on AAVE. These values will be sums or averages across all open positions.
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "user": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
# }
def get_aave_user_position_summary(json_data: str = "{}") -> Dict[str, Any]:
    """ Fetches Aave user position summary using JSON input. """
    from compass.api_client.models.aave_get_user_position_summary import AaveGetUserPositionSummary
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        aave_user_position_instance = AaveGetUserPositionSummary.from_json(json_data)

        # 🚀 **Make the actual API request to fetch user position summary**
        api_response = api_instance.process_request_v0_aave_user_position_summary_get_post(aave_user_position_instance)

        # Convert response to dictionary
        return {"position_data": api_response.to_dict()}

    except Exception as e:
        return {"success": False, "error": str(e)}
# Get the user's position for a specific token.
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "user": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
#   "asset": "1INCH"
# }
def get_aave_user_position_per_token(json_data: str = "{}") -> Dict[str, Any]:
    """ Fetches Aave user position per token using JSON input. """
    from compass.api_client.models.aave_get_user_position_per_token import AaveGetUserPositionPerToken
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        aave_user_position_token_instance = AaveGetUserPositionPerToken.from_json(json_data)

        # 🚀 **Make the actual API request to fetch user position per token**
        api_response = api_instance.process_request_v0_aave_user_position_per_token_get_post(aave_user_position_token_instance)

        # Convert response to dictionary
        return {"position_per_token_data": api_response.to_dict()}

    except Exception as e:
        return {"success": False, "error": str(e)}
# Gets the change in liquidity index between two blocks, therefore the amount a position will have increased or decreased over the time
# Example BODY:
# {
#   "chain": "ethereum:mainnet",
#   "start_block": 21500000,
#   "end_block": 0,
#   "asset": "1INCH"
# }
def get_aave_liquidity_change(json_data: str = "{}") -> Dict[str, Any]:
    """ Fetches Aave liquidity change data using JSON input. """
    from compass.api_client.models.aave_get_liquidity_change import AaveGetLiquidityChange
    api_instance = compass.api_client.AaveV3Api(api_client)  # Create API instance

    try:
        # Create request object from JSON input
        aave_liquidity_change_instance = AaveGetLiquidityChange.from_json(json_data)

        # 🚀 **Make the actual API request to fetch liquidity change data**
        api_response = api_instance.process_request_v0_aave_liquidity_change_get_post(aave_liquidity_change_instance)

        # Convert response to dictionary
        return {"liquidity_change_data": api_response.to_dict()}

    except Exception as e:
        return {"success": False, "error": str(e)}

def run(**kwargs) -> Dict[str, Any]:
    """
    Executes API calls based on provided keyword arguments.

    :param kwargs: Dictionary containing function names and dynamic parameters.
    :return: A dictionary containing only the relevant API response.
    """
    action = kwargs.get("action")
    json_data = kwargs.get("json_data", "{}")

    if action == "post_aave_supply":
        return post_aave_supply(json_data)

    elif action == "post_aave_borrow":
        return post_aave_borrow(json_data)

    elif action == "post_aave_repay_borrowed":
        return post_aave_repay_borrowed(json_data)

    elif action == "post_aave_withdraw_collateral":
        return post_aave_withdraw_collateral(json_data)

    elif action == "get_aave_asset_price":
        return get_aave_asset_price(json_data)

    elif action == "get_aave_user_position_summary":
        return get_aave_user_position_summary(json_data)

    elif action == "get_aave_user_position_per_token":
        return get_aave_user_position_per_token(json_data)

    elif action == "get_aave_liquidity_change":
        return get_aave_liquidity_change(json_data)

    else:
        return {"success": False, "error": "Invalid action provided"}