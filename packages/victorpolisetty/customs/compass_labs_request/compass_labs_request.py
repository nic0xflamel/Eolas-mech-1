import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_compass.toolkits import LangchainCompassToolkit

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("YOUR-GPT-KEY-HERE")
# Initialize Compass Toolkit
try:
    toolkit = LangchainCompassToolkit(compass_api_key=API_KEY)
    tools = toolkit.get_tools()
    print(tools)
except Exception as e:
    raise RuntimeError(f"Failed to initialize LangchainCompassToolkit: {e}")

# LLM setup
llm = ChatOpenAI(model="gpt-4o-mini")

# LangChain agent with all tools (no filtering)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # ✅ Supports multi-input tools
    verbose=True,
)

# Run Compass API based on natural language
def run(prompt: str) -> Dict[str, Any]:
    if not prompt or len(prompt.strip()) < 3:
        return {"error": "Prompt too short or empty."}
    try:
        result = agent.run(prompt)
        return {"result": result}
    except Exception as e:
        return {"error": f"Agent execution failed: {str(e)}"}

# CLI entry
if __name__ == "__main__":
    print("\n🧠 Compass Labs Natural Language Agent\n")
    user_input = input("Ask something: ")
    output = run(user_input)

    print("\n=== RESPONSE ===")
    if "result" in output:
        print(output["result"])
    else:
        print(f"❌ Error: {output['error']}")