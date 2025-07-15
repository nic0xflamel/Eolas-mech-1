#!/usr/bin/env python3
"""
Test file for the Intelligent DeFi Assistant
Demonstrates the ReAct agent pattern with natural language queries
"""

import json
from compass_defi import run

def test_simple_query():
    """Test a simple rate query"""
    print("=" * 60)
    print("🧠 INTELLIGENT DEFI ASSISTANT TEST")
    print("=" * 60)
    
    # Example query that would work with a valid Gemini API key
    query = "What is the current USDC lending rate on Base?"
    
    print(f"📝 Query: {query}")
    print("\n🔄 Expected Flow:")
    print("1. Gemini analyzes: 'User wants AAVE rates for USDC on Base'")
    print("2. Tool Selection: get_aave_rates")
    print("3. Parameters: {'chain': 'base:mainnet', 'token': 'USDC'}")
    print("4. API Call: compass_api.aave_v3.rate(...)")
    print("5. Response: 'USDC lending rate on Base is 3.92% APY'")
    
    # Show the tool definitions
    print("\n🛠️ Available Tools:")
    from compass_defi import DEFI_TOOLS
    for tool_name, tool_def in DEFI_TOOLS.items():
        print(f"  • {tool_name}: {tool_def['description']}")
    
    print("\n" + "=" * 60)

def test_complex_query():
    """Test a complex multi-step query"""
    print("🔥 COMPLEX QUERY EXAMPLE")
    print("=" * 60)
    
    query = "Compare USDC and WETH rates on Base, and also get the current WETH price"
    
    print(f"📝 Query: {query}")
    print("\n🔄 Expected Multi-Step Flow:")
    print("1. Gemini analyzes: 'User wants rates for both tokens plus WETH price'")
    print("2. Step 1: get_aave_rates(chain='base:mainnet', token='USDC')")
    print("3. Step 2: get_aave_rates(chain='base:mainnet', token='WETH')")
    print("4. Step 3: get_token_price(chain='base:mainnet', token='WETH')")
    print("5. Final Answer: Comprehensive comparison with rates and price")
    
    print("\n" + "=" * 60)

def test_portfolio_query():
    """Test a portfolio query"""
    print("💼 PORTFOLIO QUERY EXAMPLE")
    print("=" * 60)
    
    query = "Show me the portfolio for address 0x1234567890123456789012345678901234567890"
    
    print(f"📝 Query: {query}")
    print("\n🔄 Expected Flow:")
    print("1. Gemini extracts address from query")
    print("2. Tool Selection: get_user_portfolio")
    print("3. Parameters: {'chain': 'base:mainnet', 'address': '0x1234...'}")
    print("4. Response: Portfolio breakdown with token balances")
    
    print("\n" + "=" * 60)

def show_agent_architecture():
    """Show the agent architecture"""
    print("🏗️ REACT AGENT ARCHITECTURE")
    print("=" * 60)
    
    print("""
    ┌─────────────────┐
    │   User Query    │
    │ "USDC rate on   │
    │     Base?"      │
    └─────────┬───────┘
              │
              v
    ┌─────────────────┐
    │  Gemini Model   │
    │   Reasoning     │
    │ • Analyze query │
    │ • Select tool   │
    │ • Extract params│
    └─────────┬───────┘
              │
              v
    ┌─────────────────┐
    │  Tool Executor  │
    │ • Validate tool │
    │ • Call Compass  │
    │ • Return result │
    └─────────┬───────┘
              │
              v
    ┌─────────────────┐
    │ Response Format │
    │ • Natural lang  │
    │ • Structured    │
    │ • Show reasoning│
    └─────────────────┘
    """)
    
    print("Key Features:")
    print("✅ Natural Language Understanding")
    print("✅ Multi-step Reasoning")
    print("✅ Read-only Safety")
    print("✅ MCP-style Tool Definitions")
    print("✅ Comprehensive Error Handling")
    print("✅ Structured Response Format")
    
    print("\n" + "=" * 60)

def show_usage_examples():
    """Show usage examples"""
    print("📚 USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            "query": "What's the USDC rate on Base?",
            "expected_tool": "get_aave_rates",
            "description": "Simple rate query"
        },
        {
            "query": "Get WETH price on Ethereum",
            "expected_tool": "get_token_price", 
            "description": "Token price query"
        },
        {
            "query": "Show AAVE reserve info for USDC on Base",
            "expected_tool": "get_aave_reserve_overview",
            "description": "Reserve overview query"
        },
        {
            "query": "What's the USDC/WETH pool price on Uniswap?",
            "expected_tool": "get_uniswap_pool_price",
            "description": "Pool price query"
        },
        {
            "query": "Compare USDC rates across Base and Ethereum",
            "expected_tool": "get_aave_rates (multiple calls)",
            "description": "Multi-chain comparison"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   Query: \"{example['query']}\"")
        print(f"   Tool: {example['expected_tool']}")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    show_agent_architecture()
    test_simple_query()
    test_complex_query()
    test_portfolio_query()
    show_usage_examples()
    
    print("\n🚀 TO USE THE AGENT:")
    print("1. Get a Gemini API key from https://makersuite.google.com/app/apikey")
    print("2. Call run() with your query and API keys")
    print("3. The agent will reason through your query and provide intelligent responses!")
    
    print("\n💡 Example Python usage:")
    print("""
from compass_defi import run

result = run(
    query="What's the USDC lending rate on Base?",
    compass_api_key="your_compass_key",
    gemini_api_key="your_gemini_key"
)
print(result)
""") 