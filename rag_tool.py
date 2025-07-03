import os
import sys
import logging
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add parent directory to path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from security.pii_filter import detect_pii
from security.compliance_tagger import detect_compliance_issues

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGTool:
    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = "allyin_docs"

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        embedding = self.model.encode([query])[0].tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=0.3
        )
        return results.points

    def build_prompt(self, query: str, docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = (
                doc.payload.get("chunk_text") or
                doc.payload.get("content") or
                doc.payload.get("text") or
                doc.payload.get("document") or
                doc.payload.get("raw_text") or
                doc.payload.get("text_body") or
                ""
            ).strip()

            subject = doc.payload.get("subject", "")
            from_email = doc.payload.get("from", "")
            to_email = doc.payload.get("to", "")
            date = doc.payload.get("date", "")
            filename = doc.payload.get("filename", "unknown")
            score = doc.score

            metadata = []
            if subject:
                metadata.append(f"Subject: {subject}")
            if from_email:
                metadata.append(f"From: {from_email}")
            if to_email:
                metadata.append(f"To: {to_email}")
            if date:
                metadata.append(f"Date: {date}")

            metadata_block = "\n".join(metadata)

            context_parts.append(
                f"[Source {i}: {filename} (Score: {score:.2f})]\n"
                f"{metadata_block}\n{content}\n"
            )

        context = "\n".join(context_parts)

        return (
            "You are AllyIn Compass, a helpful enterprise assistant. "
            "Use the provided context to answer the user's question. "
            "If no relevant violations are found, explain that everything is in compliance based on available data."
            f"\n\nUse this information to answer the question:\n{context}\n\nQuestion: {query}"
        )

    def generate_answer(self, prompt: str) -> str:
        try:
            completion = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are AllyIn Compass, a helpful enterprise assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return "LLM failed to generate answer."

    def query(self, question: str) -> Dict[str, Any]:
        logger.info(f"Processing question: {question}")
        start_time = time.time()

        docs = self.search_documents(question)
        prompt = self.build_prompt(question, docs)
        answer = self.generate_answer(prompt)

        pii = []
        compliance_flags = []
        citations = []

        for doc in docs:
            content = (
                doc.payload.get("chunk_text") or
                doc.payload.get("content") or
                doc.payload.get("text") or
                doc.payload.get("document") or
                doc.payload.get("raw_text") or
                doc.payload.get("text_body") or
                ""
            ).strip()

            filename = doc.payload.get("filename", "unknown")
            preview = content[:200]

            citations.append({
                "source": filename,
                "relevance_score": round(doc.score, 3),
                "content_preview": preview + "..." if preview else "(No preview available)"
            })

            pii_result = detect_pii(content)
            compliance_result = detect_compliance_issues(content)
            if pii_result:
                pii.append({"source": filename, "details": pii_result})
            if compliance_result:
                compliance_flags.append({"source": filename, "details": compliance_result})

        response_time = round(time.time() - start_time, 2)

        return {
            "question": question,
            "answer": answer,
            "confidence": round(sum(d.score for d in docs) / len(docs), 2) if docs else 0,
            "sources": [d.payload.get("filename", "unknown") for d in docs],
            "citations": citations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_used": "gpt-3.5-turbo",
            "num_sources": len(docs),
            "response_time": response_time,
            "pii": pii,
            "compliance_flags": compliance_flags
        }

# Factory function
def create_rag_tool(config=None):
    return RAGTool()

# Optional CLI test
if __name__ == "__main__":
    rag = create_rag_tool()
    question = "What are the latest CO2 compliance violations in San Jose?"
    result = rag.query(question)
    print(json.dumps(result, indent=2))
