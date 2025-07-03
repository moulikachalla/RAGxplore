import os
import duckdb
import pandas as pd
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import OpenAI

load_dotenv()

class SQLRetriever:
    def __init__(self, db_path: str = "data/structured/allyin.duckdb", usage_logger: Optional[Any] = None):
        self.db_path = db_path
        self.usage_logger = usage_logger

        self._ensure_tables_exist()

        db_uri = f"duckdb:///{db_path}"
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = OpenAI(temperature=0)
        self.sql_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=True)

    def _ensure_tables_exist(self):
        conn = duckdb.connect(self.db_path)
        tables_needed = {
            "facilities": "data/structured/facilities.csv",
            "customers": "data/structured/customers.csv",
            "orders": "data/structured/orders.csv"
        }

        existing_tables = set(conn.execute("SHOW TABLES").fetchdf()["name"])
        for table, path in tables_needed.items():
            if table not in existing_tables and os.path.exists(path):
                df = pd.read_csv(path)
                conn.register(f"{table}_df", df)
                conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM {table}_df")
                print(f"[INFO] Auto-loaded table: {table}")
            elif table not in existing_tables:
                print(f"[WARNING] Missing file: {path} — Cannot auto-create table '{table}'")
        conn.close()

    def query(self, natural_query: str) -> Dict[str, Any]:
        try:
            result = self.sql_chain.invoke({"query": natural_query})
            if self.usage_logger:
                self.usage_logger.log_tool_call(
                    tool_name="sql_query",
                    query=natural_query,
                    result=str(result),
                    duration=0,
                    success=True
                )
            return {"success": True, "result": result, "source": "SQL Database"}
        except Exception as e:
            if self.usage_logger:
                self.usage_logger.log_tool_call(
                    tool_name="sql_query",
                    query=natural_query,
                    result=str(e),
                    duration=0,
                    success=False
                )
            return {"success": False, "error": str(e), "source": "SQL Database"}

if __name__ == "__main__":
    retriever = SQLRetriever()
    q = "List non-compliant facilities in California"
    result = retriever.query(q)
    print(result["result"] if result["success"] else result["error"])
