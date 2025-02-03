import functools

import openai
import requests
from typing import Dict, Optional, Tuple, Any, Callable
from openai import OpenAI


class LunarCrushAPI:
    """Client for interacting with the LunarCrush API"""
    BASE_URL = "https://lunarcrush.com/api4"
    
    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
    
    def get_coins_list(self) -> Dict[str, Any]:
        """Get detailed data for a specific crypto asset"""
        endpoint = f"{self.BASE_URL}/public/coins/list/v1"
        response = requests.request("GET", endpoint, headers=self.headers)
        response.raise_for_status()
        return response.json()


class APIClients:
    def __init__(self, api_keys: Any):
        self.lunarcrush_api_key = api_keys["lunarcrush"]
        self.openai_api_key = api_keys["openai"]
        
        if not all([self.lunarcrush_api_key, self.openai_api_key]):
            raise ValueError("Missing required API keys in environment variables")
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.lunarcrush_client = LunarCrushAPI(self.lunarcrush_api_key)


def extract_token_symbol(clients: APIClients, prompt: str) -> str:
    """
    Extract cryptocurrency token symbol from the given prompt using OpenAI.
    
    Args:
        clients: APIClients object containing initialized API clients
        prompt: Text containing discussion about a cryptocurrency
        
    Returns:
        str: The extracted token symbol (e.g., 'BTC', 'ETH')
    """
    try:
        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", 
                 "content": "You are a cryptocurrency token symbol extractor. "
                           "Extract and return ONLY the token symbol mentioned in the text. "
                           "If multiple symbols are found, return the most relevant one. "
                           "Return in uppercase."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Error extracting token symbol: {str(e)}")


def get_coin_data(lunar_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Find coin data for a specific symbol in LunarCrush API response.
    
    Args:
        lunar_data: List of coin data from LunarCrush API
        symbol: Token symbol to search for (e.g., 'BTC')
        
    Returns:
        Dict containing the coin data or empty dict if not found
    """
    # Normalize symbol for comparison
    symbol = symbol.upper()
    
    # Search through the data array for matching symbol
    for coin in lunar_data:
        if coin.get('symbol') == symbol:
            coin_data = {
                'id': coin.get('id'),
                'symbol': coin.get('symbol'),
                'name': coin.get('name'),
                'price': coin.get('price'),
                'price_btc': coin.get('price_btc'),
                'volume_24h': coin.get('volume_24h'),
                'volatility': coin.get('volatility'),
                'circulating_supply': coin.get('circulating_supply'),
                'max_supply': coin.get('max_supply'),
                'percent_change_1h': coin.get('percent_change_1h'),
                'percent_change_24h': coin.get('percent_change_24h'),
                'percent_change_7d': coin.get('percent_change_7d'),
                'percent_change_30d': coin.get('percent_change_30d'),
                'market_cap': coin.get('market_cap'),
                'market_cap_rank': coin.get('market_cap_rank'),
                'interactions_24h': coin.get('interactions_24h'),
                'social_volume_24h': coin.get('social_volume_24h'),
                'social_dominance': coin.get('social_dominance'),
                'market_dominance': coin.get('market_dominance'),
                'market_dominance_prev': coin.get('market_dominance_prev'),
                'galaxy_score': coin.get('galaxy_score'),
                'galaxy_score_previous': coin.get('galaxy_score_previous'),
                'alt_rank': coin.get('alt_rank'),
                'alt_rank_previous': coin.get('alt_rank_previous'),
                'sentiment': coin.get('sentiment'),
                'categories': coin.get('categories'),
                'blockchains': coin.get('blockchains'),
                'last_updated_price': coin.get('last_updated_price'),
                'last_updated_price_by': coin.get('last_updated_price_by'),
                'topic': coin.get('topic'),
                'logo': coin.get('logo')
            }
            return coin_data
            
    return {}

def get_sentiment_analysis(clients: APIClients, coin_data: Dict[str, Any]) -> Optional[str]:
    """Generate a sentiment analysis for a cryptocurrency using market and social data.
    
    Args:
        clients: APIClients instance containing OpenAI client
        coin_data: Dictionary containing coin metrics and social data
        
    Returns:
        A string containing the sentiment analysis, or None if analysis fails
    """
    if not coin_data:
        return None
        
    try:
        # Construct prompt with relevant metrics
        prompt = f"""Analyze the market sentiment for {coin_data['name']} ({coin_data['symbol']}) based on the following metrics:

Market Metrics:
- Price change 24h: {coin_data['percent_change_24h']}%
- Price change 7d: {coin_data['percent_change_7d']}%
- Market dominance: {coin_data['market_dominance']}%
- Market cap rank: #{coin_data['market_cap_rank']}
- Volatility: {coin_data['volatility']}

Social & Sentiment Metrics:
- Galaxy Score (0-100): {coin_data['galaxy_score']} (previous: {coin_data['galaxy_score_previous']})
- Social dominance: {coin_data['social_dominance']}
- 24h social interactions: {coin_data['interactions_24h']}
- 24h social volume: {coin_data['social_volume_24h']}
- Overall sentiment: {coin_data['sentiment']}

Provide a concise analysis of the current market sentiment based on these metrics.
"""

        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating sentiment analysis: {e}")
        return None


MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run sentiment analysis and return structured response."""
    try:
        clients = APIClients(api_keys)
        token_symbol = extract_token_symbol(clients=clients, prompt=prompt)

        if not token_symbol:
            return "No cryptocurrency token found in prompt", "", None, None

        lunar_list_data = clients.lunarcrush_client.get_coins_list()
        if not lunar_list_data.get('data'):
            return f"No data available from LunarCrush API", "", None, None
        
        coin_data = get_coin_data(lunar_data=lunar_list_data['data'], symbol=token_symbol)
        if not coin_data:
            return f"No data found for token {token_symbol}", "", None, None
        
        sentiment_analysis = get_sentiment_analysis(clients=clients, coin_data=coin_data)
        if not sentiment_analysis:
            return f"Failed to generate sentiment analysis for {token_symbol}", "", coin_data, None

        return sentiment_analysis, "", coin_data, None

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return str(e), "", None, None
