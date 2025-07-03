# tools/rag_transformer.py

import re
from typing import Any

class RAGTool:
    """
    This class transforms user queries into more precise or structured prompts
    for the agent. It's a simple preprocessing utility.
    """

    def __init__(self):
        pass

    def transform_query(self, question: str) -> str:
        """
        Transform the question to enhance retrieval or formatting.

        Args:
            question (str): The user's input question.

        Returns:
            str: The transformed or cleaned query.
        """
        cleaned = question.strip()

        # Optional: Add basic normalizations or prompt enhancements here
        cleaned = re.sub(r"\s+", " ", cleaned)  # remove excess whitespace

        return cleaned
