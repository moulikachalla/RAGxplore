from langchain.tools import BaseTool
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from pydantic import Field
from typing import Optional, Any
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "Search documents using semantic similarity. Use for finding topics/concepts from uploaded docs."

    usage_logger: Optional[Any] = Field(default=None, exclude=True)
    model: Optional[Any] = Field(default=None, exclude=True)
    client: Optional[Any] = Field(default=None, exclude=True)
    collection_name: str = Field(default="allyin_docs", exclude=True)

    def __init__(self, usage_logger: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.usage_logger = usage_logger

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.client = QdrantClient("localhost", port=6333)
            self._check_or_create_collection()
        except Exception as e:
            logger.warning(f"Failed to initialize vector search tool: {e}")
            self.model = None
            self.client = None

    def _check_or_create_collection(self):
        """Ensure the collection exists, or create it with dummy config."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            logger.info(f"Creating missing collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def _run(self, query: str) -> str:
        """Perform semantic search and return formatted results."""
        start_time = time.time()

        if not self.model or not self.client:
            error_msg = "Vector search not available. Ensure Qdrant and SentenceTransformer are initialized."
            if self.usage_logger:
                self.usage_logger.log_tool_call("vector_search", query, error_msg, time.time() - start_time, False)
            return error_msg

        try:
            query_embedding = self.model.encode([query])[0]
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=3,
                with_payload=True
            )

            if not results:
                result_text = "No relevant documents found."
            else:
                result_text = "Top relevant chunks:\n\n"
                for i, r in enumerate(results, 1):
                    chunk = r.payload.get("chunk_text") or r.payload.get("text", "")
                    filename = r.payload.get("filename", "unknown")
                    chunk_index = r.payload.get("chunk_index", "N/A")
                    result_text += f"{i}. [File: {filename}] (Chunk {chunk_index})\n{chunk[:300]}...\n\n"

            if self.usage_logger:
                self.usage_logger.log_tool_call("vector_search", query, result_text, time.time() - start_time, True)

            return result_text

        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            if self.usage_logger:
                self.usage_logger.log_tool_call("vector_search", query, error_msg, time.time() - start_time, False)
            return error_msg


# Run standalone test
if __name__ == "__main__":
    tool = VectorSearchTool()
    print(tool._run("Find documents about CO2 compliance in San Jose"))
