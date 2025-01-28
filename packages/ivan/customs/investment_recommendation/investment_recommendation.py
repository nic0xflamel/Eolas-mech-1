import functools
import openai
import requests
from typing import Dict, Optional, Tuple, Any, Callable, List
from openai import OpenAI


class LunarCrushAPI:
    """Client for interacting with the LunarCrush API"""
    BASE_URL = "https://lunarcrush.com/api4"
    
    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
    
    def get_trending_topics(self) -> Dict[str, Any]:
        """Get trending topics and their associated metrics from LunarCrush"""
        endpoint = f"{self.BASE_URL}/public/topics/list/v1"
        response = requests.request("GET", endpoint, headers=self.headers)
        response.raise_for_status()
        return response.json()


class APIClients:
    def __init__(self, api_keys: Any):
        self.lunarcrush_api_key = api_keys["lunarcrush"]
        self.perplexity_api_key = api_keys["perplexity"]
        
        if not all([self.lunarcrush_api_key, self.perplexity_api_key]):
            raise ValueError("Missing required API keys in environment variables")
            
        self.lunarcrush_client = LunarCrushAPI(self.lunarcrush_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )


def get_top_3_trending_topics(trending_topics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return trending_topics['data'][:3]

def get_rec_number(prompt: str) -> int:
    """
    Parse the prompt to determine number of cryptocurrency recommendations requested.
    If no specific number is found, defaults to 3.
    
    Args:
        prompt: User input string that may contain a number
        
    Returns:
        Integer representing number of recommendations to provide
    """
    # Common number words mapping
    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Convert prompt to lowercase for matching
    prompt_lower = prompt.lower()
    
    # First try to find written number words
    for word, num in number_words.items():
        if word in prompt_lower:
            return num
            
    # Then look for numeric digits
    import re
    numbers = re.findall(r'\d+', prompt)
    if numbers:
        # Take first number found, limit to reasonable range
        return min(max(int(numbers[0]), 1), 10)
        
    # Default to 3 if no number found
    return 3


def get_investment_recommendation(clients: APIClients, rec_count: int, top_3_trending_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get cryptocurrency investment recommendations based on trending topics.
    
    Args:
        clients: APIClients instance containing OpenAI and LunarCrush clients
        rec_count: Number of cryptocurrency recommendations to return
        top_3_trending_topics: List of top 3 trending topics from LunarCrush
        
    Returns:
        Dictionary containing recommendations for cryptocurrencies with their
        descriptions and related trending topics
    """
    # Extract just the topic names
    topics = [topic['topic'] for topic in top_3_trending_topics]
    
    # Create prompt to identify relevant cryptocurrencies based on topics
    coins_prompt = f"""Here are some trending cryptocurrency topics:
    {topics}
    
    Based on these trending topics, identify the {rec_count} most relevant but lesser-known cryptocurrencies. 
    Focus on promising projects that are not mainstream like Bitcoin, Ethereum, or Dogecoin.
    Look for emerging or undervalued cryptocurrencies that align well with these trends but haven't received as much attention.

    For each cryptocurrency you identify, provide:
    1. The name of the cryptocurrency with its symbol in parenthesis
    2. A comprehensive description of the project, including why it's uniquely positioned for these trends
    
    Format as a structured list with clear headings. Have your response start with the word 'Here are the recommendations based on the trending topics from the past 24 hours:'
    
    Finally, in your response do not mention that you are excluding mainstream coins like Bitcoin, Ethereum, or Dogecoin. Nor should you mention which are the trending topics."""
    
    recs_response = clients.perplexity_client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online", 
        messages=[{"role": "user", "content": coins_prompt}]
    )
    recommendations = recs_response.choices[0].message.content
    
    return recommendations


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

        trending_topics = clients.lunarcrush_client.get_trending_topics()

        top_3_trending_topics = get_top_3_trending_topics(trending_topics)

        reccomendation_number = get_rec_number(prompt)

        investment_recommendation = get_investment_recommendation(clients=clients, rec_count=reccomendation_number, 
                                                                  top_3_trending_topics=top_3_trending_topics)

        return investment_recommendation, "", top_3_trending_topics, None

    except Exception as e:
        print(f"Error in investment recommendation: {str(e)}")
        return str(e), "", None, None