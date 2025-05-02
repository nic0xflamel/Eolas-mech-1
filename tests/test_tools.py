# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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
"""This module contains tool tests."""
from typing import List, Any

from packages.moserah.customs.trump_news import trump_news
from packages.uzo.customs.grid_analyser import grid_analyser
from packages.uzo.customs.trump_news_agent import trump_news_agent
from packages.jisong.customs.nft_appraisal_skill import nft_appraisal_skill
from packages.jayshree.customs.grid_pair_screener import grid_pair_screener
from packages.valory.skills.task_execution.utils.apis import KeyChain
from packages.valory.skills.task_execution.utils.benchmarks import TokenCounterCallback
from tests.constants import (
    OPENAI_SECRET_KEY,
    STABILITY_API_KEY,
    GOOGLE_API_KEY,
    GOOGLE_ENGINE_ID,
    CLAUDE_API_KEY,
    REPLICATE_API_KEY,
    NEWS_API_KEY,
    OPENROUTER_API_KEY,
    GNOSIS_RPC_URL,
    GEMINI_API_KEY, 
    SERPER_API_KEY,
    DUNE_API_KEY,
    ALCHEMY_API_KEY,
    EXCHANGE_API_KEY,
    EXCHANGE_API_SECRET
)


class BaseToolTest:
    """Base tool test class."""

    keys = KeyChain(
        {
            "openai": [OPENAI_SECRET_KEY],
            "stabilityai": [STABILITY_API_KEY],
            "google_api_key": [GOOGLE_API_KEY],
            "google_engine_id": [GOOGLE_ENGINE_ID],
            "anthropic": [CLAUDE_API_KEY],
            "replicate": [REPLICATE_API_KEY],
            "news_api": [NEWS_API_KEY],
            "openrouter": [OPENROUTER_API_KEY],
            "gnosis_rpc_url": [GNOSIS_RPC_URL],
            "gemini": [GEMINI_API_KEY],
            "serperapi": [SERPER_API_KEY],
            "dune": [DUNE_API_KEY],
            "alchemy": [ALCHEMY_API_KEY]
        }
    )
    models: List = [None]
    tools: List[str]
    prompts: List[str]
    tool_module: Any = None
    tool_callable: str = "run"

    def _validate_response(self, response: Any) -> None:
        """Validate response."""
        assert type(response) == tuple, "Response of the tool must be a tuple."
        assert len(response) == 5, "Response must have 5 elements."
        assert type(response[0]) == str, "Response[0] must be a string."
        assert type(response[1]) == str, "Response[1] must be a string."
        assert (
            type(response[2]) == dict or response[2] is None
        ), "Response[2] must be a dictionary or None."
        assert (
            type(response[3]) == TokenCounterCallback or response[3] is None
        ), "Response[3] must be a TokenCounterCallback or None."
        assert type(response[4]) == KeyChain, "Response[4] must be a KeyChain object."

    def test_run(self) -> None:
        """Test run method."""
        assert self.tools, "Tools must be provided."
        assert self.prompts, "Prompts must be provided."
        assert self.tool_module, "Callable function must be provided."

        for model in self.models:
            for tool in self.tools:
                for prompt in self.prompts:
                    kwargs = dict(
                        prompt=prompt,
                        tool=tool,
                        api_keys=self.keys,
                        counter_callback=TokenCounterCallback(),
                        model=model,
                    )
                    func = getattr(self.tool_module, self.tool_callable)
                    response = func(**kwargs)
                    self._validate_response(response)
                    print("response: ", response)


class TestTrumpNews(BaseToolTest):
    """Test Trump News."""

    tools = ["TOOLS"]
    models = ["MODELS"]
    prompts = [
        "What's the latest on Trump's legal issues?"
    ]
    tool_module = trump_news

class TestNFTAppraisalSkill(BaseToolTest):
    """Test Trump News."""

    tools = ["TOOLS"]
    models = ["MODELS"]
    prompts = [
        "tell me about 0xed5af388653567af2f388e6224dc7c4b3241c544, which is on eth."
    ]
    tool_module = nft_appraisal_skill

class TestGridPairScreener(BaseToolTest):
    """Test Grid Pair Screener."""

    tools = ["TOOLS"]
    models = ["MODELS"]
    prompts = [
        "tell me about 0xed5af388653567af2f388e6224dc7c4b3241c544, which is on eth."
    ]
    tool_module = grid_pair_screener

class TestTrumpNewsAgent(BaseToolTest):
    """Test Trump News Agent."""

    tools = ["TOOLS"]
    models = ["gpt-4o-mini"]
    prompts = [
        "what is the latest trump news"
    ]
    tool_module = trump_news_agent

class TestGridTradingAnalyzer(BaseToolTest):
    """Test Grid Trading Analyzer."""

    tools = ["TOOLS"]
    models = ["MODELS"]
    prompts = [
        ""
    ]
    tool_module = grid_analyser

    def test_run(self) -> None:
        """Test run method."""
        assert self.tools, "Tools must be provided."
        assert self.prompts, "Prompts must be provided."
        assert self.tool_module, "Callable function must be provided."

        for model in self.models:
            for tool in self.tools:
                for prompt in self.prompts:
                    kwargs = dict(
                        prompt=prompt,
                        tool=tool,
                        api_keys=self.keys,
                        counter_callback=TokenCounterCallback(),
                        model=model,
                        command="scan",
                        exchange="coinbase",
                        exchange_api_key=EXCHANGE_API_KEY,
                        exchange_api_secret=EXCHANGE_API_SECRET,
                        quote_currency="USDT",
                        symbols=["BTC/USDT"],
                        output_format="dict"
                    )
                    func = getattr(self.tool_module, self.tool_callable)
                    response = func(**kwargs)
                    self._validate_response(response)
                    print("response: ", response)
