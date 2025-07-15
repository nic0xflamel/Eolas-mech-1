# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 nic_f
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Intelligent DeFi assistant using LangGraph and Compass API."""

import functools
import json
import os
import inspect
from typing import Annotated, Any, Sequence, TypedDict, Tuple, Optional, Dict

from compass_api_sdk import CompassAPI
from google.api_core import exceptions as google_exceptions
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# System prompt as a constant for clarity
SYSTEM_PROMPT = """You are a DeFi assistant that can ONLY answer questions using the provided tools. You have NO access to external knowledge or real-time data beyond these tools.

CRITICAL CONSTRAINTS:
- You MUST use the provided tools to gather ALL information
- You CANNOT answer from your own knowledge or training data
- You CANNOT make up data or cite external sources like CoinGecko, CoinMarketCap, etc.
- If the tools cannot provide the requested information, you must say so clearly

WORKFLOW:
1. Analyze the user's query to determine which tool(s) to use
2. Call the appropriate tool(s) to gather the required data
3. Use the final_response tool to format your answer with the exact data from the tools

PARAMETERS:
- Chain format: "base:mainnet", "ethereum:mainnet", "arbitrum:mainnet"
- Common tokens: USDC, WETH, USDT, DAI, etc.
- If no chain is specified, default to "ethereum:mainnet"

MANDATORY: Always use the final_response tool to provide your final answer with standardized format.

Remember: You are a tool-based assistant. You cannot know anything beyond what the tools tell you."""


# The decorator for handling API key rotation and retries
def with_key_rotation(func):
    """A decorator to handle API key rotation and retries for rate limit errors."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        api_keys = kwargs.get("api_keys")
        if not api_keys:
            return "Error: api_keys object not provided."

        try:
            # The tool example uses a 'max_retries' method on the keychain.
            # We will replicate this logic by using the number of available keys as the retry count.
            retries_left = {
                "gemini_api": len(api_keys.get("gemini_api", [])),
                "compass_api": len(api_keys.get("compass_api", [])),
            }
        except Exception:
            # Fallback if api_keys is not a list-based keychain
            retries_left = {"gemini_api": 1, "compass_api": 1}

        def execute():
            """Execute the function with retry logic."""
            try:
                return func(*args, **kwargs)
            except google_exceptions.ResourceExhausted as e:
                service = "gemini_api"
                if retries_left.get(service, 0) > 0:
                    print(f"Gemini API rate limit exceeded. Rotating key and retrying.")
                    retries_left[service] -= 1
                    api_keys.rotate(service)
                    return execute()
                return f"Error: Gemini API rate limit exceeded. No more keys to try. Details: {e}"
            except Exception as e:
                # Generic exception for Compass API or other issues
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    service = "compass_api"
                    if retries_left.get(service, 0) > 0:
                        print(f"Compass API rate limit likely exceeded. Rotating key and retrying.")
                        retries_left[service] -= 1
                        api_keys.rotate(service)
                        return execute()
                    return f"Error: Compass API rate limit exceeded. No more keys to try. Details: {e}"
                
                # For any other exception, return the error string.
                return f"An unexpected error occurred: {str(e)}"

        return execute()

    return wrapper


# Agent State following LangGraph pattern exactly
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int
    compass_api_key: str


# Tool Input Schemas
class SearchInput(BaseModel):
    chain: str = Field(description="Blockchain network (base:mainnet, ethereum:mainnet, arbitrum:mainnet)")
    token: str = Field(description="Token symbol (e.g., USDC, WETH, USDT)")

class PortfolioInput(BaseModel):
    chain: str = Field(description="Blockchain network (base:mainnet, ethereum:mainnet, arbitrum:mainnet)")
    address: str = Field(description="User's wallet address")

class PoolInput(BaseModel):
    chain: str = Field(description="Blockchain network (base:mainnet, ethereum:mainnet, arbitrum:mainnet)")
    token_in: str = Field(description="Input token symbol")
    token_out: str = Field(description="Output token symbol")

# LangGraph Native Structured Response Schema
class DeFiResponse(BaseModel):
    """Structured response for DeFi queries with standardized format."""
    answer: str = Field(description="The main answer to the user's query")
    data_source: str = Field(description="Source of the data (e.g., 'AAVE V3 on Base', 'Uniswap V3', etc.)")
    blockchain: str = Field(description="Blockchain network name (e.g., 'Base', 'Ethereum', 'Arbitrum')")
    additional_info: str = Field(description="Any additional context or helpful information", default="")

# DeFi Tools using dependency injection for the API key, avoiding globals
@tool("get_aave_rates", args_schema=SearchInput, return_direct=False)
def get_aave_rates(chain: str, token: str, compass_api_key: str):
    """Get AAVE V3 lending and borrowing rates for a specific token on a blockchain."""
    try:
        with CompassAPI(api_key_auth=compass_api_key) as compass_api:
            response = compass_api.aave_v3.rate(chain=chain, token=token)
            data = response.model_dump()
            return f"AAVE rates for {token} on {chain}: Supply APY: {float(data.get('supply_apy_variable_rate', 0))*100:.2f}%, Borrow APY: {float(data.get('borrow_apy_variable_rate', 0))*100:.2f}%"
    except Exception as e:
        return f"Error getting AAVE rates: {str(e)}"

@tool("get_token_price", args_schema=SearchInput, return_direct=False)
def get_token_price(chain: str, token: str, compass_api_key: str):
    """Get current price of a token in USD."""
    try:
        with CompassAPI(api_key_auth=compass_api_key) as compass_api:
            response = compass_api.token.price(chain=chain, token=token)
            data = response.model_dump()
            price = data.get('price', 'N/A')
            return f"{token} price on {chain}: ${price}"
    except Exception as e:
        return f"Error getting token price: {str(e)}"

@tool("get_aave_reserve_overview", args_schema=SearchInput, return_direct=False)
def get_aave_reserve_overview(chain: str, token: str, compass_api_key: str):
    """Get AAVE reserve overview including TVL, utilization ratio, and total borrowed."""
    try:
        with CompassAPI(api_key_auth=compass_api_key) as compass_api:
            response = compass_api.aave_v3.reserve_overview(chain=chain, token=token)
            data = response.model_dump()
            tvl = data.get('tvl', 0)
            utilization = data.get('utilization_ratio', 0)
            borrowed = data.get('total_borrowed', 0)
            return f"AAVE {token} reserve on {chain}: TVL: ${tvl:,.0f}, Utilization: {utilization:.1%}, Total Borrowed: ${borrowed:,.0f}"
    except Exception as e:
        return f"Error getting reserve overview: {str(e)}"

@tool("get_user_portfolio", args_schema=PortfolioInput, return_direct=False)
def get_user_portfolio(chain: str, address: str, compass_api_key: str):
    """Get user's DeFi portfolio across protocols."""
    try:
        with CompassAPI(api_key_auth=compass_api_key) as compass_api:
            response = compass_api.universal.portfolio(chain=chain, address=address)
            data = response.model_dump()
            return f"Portfolio for {address[:10]}... on {chain}: {json.dumps(data, indent=2)}"
    except Exception as e:
        return f"Error getting portfolio: {str(e)}"

@tool("get_uniswap_pool_price", args_schema=PoolInput, return_direct=False)
def get_uniswap_pool_price(chain: str, token_in: str, token_out: str, compass_api_key: str):
    """Get Uniswap V3 pool price between two tokens."""
    try:
        with CompassAPI(api_key_auth=compass_api_key) as compass_api:
            response = compass_api.uniswap_v3.pool_price(
                chain=chain,
                token_in_token=token_in,
                token_out_token=token_out,
                fee="3000"
            )
            data = response.model_dump()
            return f"Uniswap {token_in}/{token_out} pool price on {chain}: {json.dumps(data, indent=2)}"
    except Exception as e:
        return f"Error getting pool price: {str(e)}"

# LangGraph Native Final Response Tool (following the exact documentation pattern)
@tool("final_response", args_schema=DeFiResponse, return_direct=False)
def final_response_tool(answer: str, data_source: str, blockchain: str, additional_info: str = ""):
    """Always respond to the user using this tool with standardized format."""
    return "Response formatted and ready for user"

# Tools list
tools = [
    get_aave_rates,
    get_token_price,
    get_aave_reserve_overview,
    get_user_portfolio,
    get_uniswap_pool_price,
    final_response_tool  # Add the final response tool
]

# Create tools dictionary
tools_by_name = {tool.name: tool for tool in tools}

# Refactored tool node to dynamically inject the API key
def call_tool(state: AgentState):
    """Execute tools based on tool calls in the last message."""
    compass_api_key = state.get("compass_api_key")
    if not compass_api_key:
        raise ValueError("Compass API key not found in agent state.")

    outputs = []
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        tool_to_call = tools_by_name[tool_call["name"]]
        tool_args = tool_call["args"]

        # Inspect the tool's signature and add the key if the tool requires it.
        # This avoids needing to pass the key for tools that don't use it.
        sig = inspect.signature(tool_to_call.func)
        if "compass_api_key" in sig.parameters:
            tool_args["compass_api_key"] = compass_api_key

        # Get the tool by name
        tool_result = tool_to_call.invoke(tool_args)
        outputs.append(
            ToolMessage(
                content=str(tool_result),  # Ensure content is a string
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    """Determine whether to continue with tool calls or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are no tool calls, we finish
    if not last_message.tool_calls:
        return "end"
    
    # If the last tool call is the final_response tool, we finish
    if last_message.tool_calls[0]["name"] == "final_response":
        return "end"
    
    # Otherwise continue with tools
    return "continue"

# Create the graph following the exact Google AI pattern
def create_defi_agent(gemini_api_key: str, model_name: str):
    """Create the DeFi ReAct agent graph."""

    # Create LLM instance once, following Google AI example
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=1.0,
        max_retries=2,
        google_api_key=gemini_api_key,
    )
    # Bind tools to the model
    model = llm.bind_tools(tools)

    def call_model(state: AgentState, config: RunnableConfig):
        """Call the model with the current state and increment step count."""
        # Invoke the model with the system prompt and the messages
        response = model.invoke(state["messages"], config)
        # We return a list, because this will get added to the existing messages state using the add_messages reducer
        return {
            "messages": [response],
            "number_of_steps": state.get("number_of_steps", 0) + 1,
        }

    # Define a new graph with our state
    workflow = StateGraph(AgentState)

    # 1. Add our nodes
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)

    # 2. Set the entrypoint as `llm`, this is the first node called
    workflow.set_entry_point("llm")

    # 3. Add a conditional edge after the `llm` node is called.
    workflow.add_conditional_edges(
        # Edge is used after the `llm` node is called.
        "llm",
        # The function that will determine which node is called next.
        should_continue,
        # Mapping for where to go next, keys are strings from the function return, and the values are other nodes.
        # END is a special node marking that the graph is finished.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # 4. Add a normal edge after `tools` is called, `llm` node is called next.
    workflow.add_edge("tools", "llm")

    # Now we can compile the graph
    return workflow.compile()


# Main entry point, adapted for the mech framework
@with_key_rotation
def run(**kwargs: Any) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """The main entry point for the intelligent DeFi assistant."""

    # Extract parameters from kwargs, following the mech tool pattern
    user_query = kwargs.get("prompt")  # The mech framework uses 'prompt' for the main query
    model_name = kwargs.get("model", "gemini-2.5-pro-latest")  # Allow model to be specified
    api_keys = kwargs.get("api_keys")  # The mech framework provides a keychain object
    counter_callback = kwargs.get("counter_callback")

    # Validate inputs
    if not user_query:
        return "Error: No query provided in 'prompt'. Please provide a 'prompt' parameter.", "", None, counter_callback

    if not api_keys:
        return "Error: No api_keys object provided. Please provide the 'api_keys' keychain.", "", None, counter_callback

    gemini_api_key = api_keys.get("gemini_api")
    compass_api_key = api_keys.get("compass_api")

    if not gemini_api_key:
        return "Error: 'gemini_api' key not found in api_keys object.", "", None, counter_callback
    if not compass_api_key:
        return "Error: 'compass_api' key not found in api_keys object.", "", None, counter_callback

    prompt_string = ""
    structured_data = None
    try:
        # Create the agent
        graph = create_defi_agent(gemini_api_key, model_name)

        # Create our initial message dictionary with system prompt
        inputs = {
            "messages": [
                ("system", SYSTEM_PROMPT),
                ("user", user_query),
            ],
            "number_of_steps": 0,
            "compass_api_key": compass_api_key,
        }
        prompt_string = json.dumps(inputs)


        # Configuration for the graph
        config = {"configurable": {}}

        # Call our graph with streaming to see the steps
        final_state = None
        for state in graph.stream(inputs, config=config, stream_mode="values"):
            final_state = state

        # Extract the final response from the final_response tool call
        if final_state and "messages" in final_state:
            for message in reversed(final_state["messages"]):
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call["name"] == "final_response":
                            structured_data = tool_call["args"]
                            # Format the structured response
                            formatted_response = f"""{structured_data['answer']}

Source: {structured_data['data_source']} on {structured_data['blockchain']}"""
                            if structured_data.get('additional_info'):
                                formatted_response += f"\n\nAdditional Info: {structured_data['additional_info']}"
                            return formatted_response, prompt_string, structured_data, counter_callback

            # Fallback to last message if no final_response tool found
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                return last_message.content, prompt_string, structured_data, counter_callback
            return str(last_message), prompt_string, structured_data, counter_callback
        
        return "No response generated", prompt_string, structured_data, counter_callback

    except Exception as e:
        return f"Error: {str(e)}", prompt_string, structured_data, counter_callback 