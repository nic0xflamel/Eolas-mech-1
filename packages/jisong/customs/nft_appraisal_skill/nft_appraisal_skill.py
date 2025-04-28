from langchain_openai import ChatOpenAI
from typing import Any, Tuple, Dict, Optional, Callable
import requests
import functools
import os

alchemy_networks = {
    "ethereum": "https://eth-mainnet.g.alchemy.com",
    "base": "https://base-mainnet.g.alchemy.com",
}


# Function to extract contract address and chain from the prompt
def extract_contract_address(llm: Any, prompt: str) -> Tuple[str, str]:
    prompt = f"""
        From given prompt, extract the contract address and chain of the NFT.
        If the prompt does not contain the contract address and chain, ask the user for it.
        print the result in the following format:
        contractaddress,chain
        such as 0xed5af388653567af2f388e6224dc7c4b3241c544,ethereum
        the given prompt is: {prompt}
    """

    completion = str(llm.invoke(prompt).content)

    contract_address, chain = completion.split(",")

    return contract_address, chain


# Use alchemy api to get NFT metadata
def getNFTMetadata(contract_address: str, chain: str, api_key: str) -> dict:
    url = (
        f"{alchemy_networks[chain]}/nft/v3/{api_key}/getContractMetadata?contractAddress="
        + contract_address
    )

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers).json()

    return response


def getNFTAttributeSummary(contract_address: str, chain: str, api_key: str) -> dict:
    url = (
        f"{alchemy_networks[chain]}/nft/v3/{api_key}/summarizeNFTAttributes?contractAddress="
        + contract_address
    )

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers).json()

    return response


def getNFTRecentSales(contract_address: str, chain: str, api_key: str) -> dict:
    url = (
        f"{alchemy_networks[chain]}/nft/v3/{api_key}/getNFTSales?fromBlock=0&toBlock=latest&order=asc&limit=100&contractAddress="
        + contract_address
    )

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers).json()

    return response


MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except Exception as e:
                service = e.__class__.__name__.lower()
                if (
                    hasattr(e, "status_code") and e.status_code == 429
                ):  # If rate limit exceeded
                    if retries_left.get(service, 0) <= 0:
                        raise e
                    retries_left[service] -= 1
                    api_keys.rotate(service)
                    return execute()

                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


# Receive a prompt and api keys, and return the response, which will praise the NFT
@with_key_rotation
def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    try:
        asking_prompt = kwargs["prompt"]  # `prompt` argument name is for compatibility with the original resolver.
        api_keys = kwargs["api_keys"]
        openai_api_key = api_keys["openai"]
        alchemy_api_key = api_keys["alchemy"]

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=openai_api_key,
        )

        contract_address, chain = extract_contract_address(llm, asking_prompt)

        response = getNFTMetadata(contract_address, chain, alchemy_api_key)
        attribute_response = getNFTAttributeSummary(
            contract_address, chain, alchemy_api_key
        )
        sales_response = getNFTRecentSales(contract_address, chain, alchemy_api_key)

        if "error" in response:
            return [response["error"]]

        name = response["name"]
        symbol = response["symbol"]
        floorPrice = response["openSeaMetadata"]["floorPrice"]
        totalSupply = response["totalSupply"]

        if "nftSales" in sales_response:
            sales_count = len(sales_response["nftSales"])
        else:
            sales_count = 0

        if "summary" in attribute_response:
            trait_count = len(attribute_response["summary"])
        else:
            trait_count = 0

        prompt = f"""
            You are an ai agent who specializes in appraising NFTs, you should make an appraisal of the following NFT user asked,
            based on analyzing market trends, rarity, historical sales, and floor prices.
            If the user does not provide contract address and chain, you should ask for it.
            In your answer, you should provide a detailed explanation of how you arrived at the appraisal value.
            In your answer, you should provide a floor price of the nft
            The contract address is {contract_address} and chain is {chain}
            The name of the NFT is {name} and the symbol is {symbol}
            The floor price of the NFT is {floorPrice}
            The total supply of the NFT is {totalSupply}

            This collection has more than {sales_count} sales and {trait_count} traits.

            Be sure to include the provided numbers so that the user knows you have done your research.

            The appraisal must be in form of
            'The NFT you asked is [name] and the symbol is [symbol]. The floor price of the NFT is [floorPrice].'
        """

        completion = str(llm.invoke(prompt).content)

        return str(completion), "", None, None

    except Exception as e:
        return f"An error occurred: {str(e)}", "", None, None
