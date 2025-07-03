# agents/multi_tool_agent.py

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
import argparse

# Add src directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.callbacks.manager import get_openai_callback

from src.retrievers.sql_retriever import SQLRetriever
from src.retrievers.vector_retriever import VectorSearchTool
from src.retrievers.graph_retriever import GraphRetriever
from tools.rag_transformer import RAGTool  # ✅ This is the correct new file

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllyInCompassAgent:
    def __init__(self, openai_api_key: str = None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY in environment or init parameter")

        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.sql_tool = SQLRetriever()
        self.vector_tool = VectorSearchTool()
        self.graph_retriever = GraphRetriever()

        tools = [
            Tool(
                name="sql_query",
                func=self.sql_tool.query,
                description="Query structured tabular data about customers or facilities."
            ),
            self.vector_tool,
            Tool(
                name="graph_search",
                func=self._graph_tool_wrapper,
                description="Retrieve insights from graph DB (e.g., compliance, industry links)."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3
        )

        self.transformer = RAGTool()

    def _graph_tool_wrapper(self, query: str) -> str:
        query_lower = query.lower()
        if "high risk" in query_lower:
            results = self.graph_retriever.find_high_risk_customers_in_industry("Biotech")
        elif "compliance" in query_lower or "violations" in query_lower:
            results = self.graph_retriever.find_compliance_violations()
        else:
            return "No graph handler matched. Try asking about high risk, compliance, or industry."
        return json.dumps(results, indent=2)

    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            logger.info(f"Processing question: {question}")
            transformed = self.transformer.transform_query(question)
            with get_openai_callback() as cb:
                result = self.agent.invoke({"input": transformed})
            return {
                "question": question,
                "transformed_query": transformed,
                "answer": result.get("output", result),
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "tokens": {
                    "prompt": cb.prompt_tokens,
                    "completion": cb.completion_tokens,
                    "total": cb.total_tokens,
                    "cost": cb.total_cost
                },
                "success": True
            }
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "question": question,
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

def main():
    parser = argparse.ArgumentParser(description="Run AllyIn Compass Agent queries")
    parser.add_argument("--question", type=str, help="Optional single question to run", default=None)
    args = parser.parse_args()

    agent = AllyInCompassAgent()

    if args.question:
        response = agent.query(args.question)
        print(json.dumps(response, indent=2))
    else:
        queries = [
            "Which customers are classified as high risk?",
            "Find facilities that exceed CO2 limits",
            "Show me all facilities and their compliance status",
            "Find documents about emissions compliance for our San Jose facility",
            "What customers have the highest revenue?",
            "Show me non-compliant facilities in California"
        ]

        print("=" * 80)
        print("ALLYIN COMPASS MULTI-TOOL AGENT - TESTING MODE")
        print("=" * 80)

        for i, question in enumerate(queries, 1):
            print(f"\n[TEST CASE {i}] {question}")
            print("-" * 60)
            response = agent.query(question)
            if response["success"]:
                print(f"Answer:\n{response['answer']}")
            else:
                print(f"Error:\n{response['error']}")
            print(f"Execution Time: {response['duration']:.2f}s")

            with open("agent_outputs.jsonl", "a") as f:
                f.write(json.dumps(response) + "\n")

if __name__ == "__main__":
    main()
