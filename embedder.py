import os, json, uuid, logging, gc, time
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', qdrant_host='localhost', qdrant_port=6333,
                 collection_name='allyin_docs', chunk_size=300, chunk_overlap=30):
        """
        Initialize embedding model and Qdrant connection.
        """
        self.embedding_model = SentenceTransformer(model_name)

        if torch.backends.mps.is_available():
            logger.info("MPS is available but using CPU to avoid memory crash.")
        self.embedding_model.to("cpu")

        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedding_dim = len(self.embedding_model.encode(["test"])[0])
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._setup_collection()

    def _setup_collection(self):
        """
        Create Qdrant collection if it does not exist.
        """
        collections = self.qdrant_client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)
            )

    def delete_existing_collection(self):
        """
        Delete existing collection for fresh embedding.
        """
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Could not delete collection: {e}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split document text into smaller chunks using RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    def load_documents_from_file(self, path: str) -> List[Dict[str, Any]]:
        """
        Load and parse documents from a JSONL file.
        """
        docs = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        doc = json.loads(line.strip())
                        docs.append(doc)
                        logger.info(f"Loaded: {doc.get('filename', f'line_{i}')}")
                    except json.JSONDecodeError:
                        logger.warning(f"Line {i} is not valid JSON.")
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise
        return docs

    def embed_and_store_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Chunk documents, embed them, and store in Qdrant with metadata.
        """
        total_chunks = 0

        for doc in documents:
            content = doc.get('content', '')
            if not content:
                logger.warning(f"Skipped {doc.get('filename', 'unknown')} - missing content.")
                continue

            chunks = self.chunk_text(content)
            logger.info(f"Creating {len(chunks)} chunks from {doc.get('filename', 'unknown')}")

            batch = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                try:
                    embedding = self.embedding_model.encode([chunk], show_progress_bar=False)[0]
                except Exception as e:
                    logger.error(f"Embedding failed for chunk {i}: {e}")
                    continue

                metadata = {
                    "document_id": doc.get('id', str(uuid.uuid4())),
                    "filename": doc.get('filename', 'unknown'),
                    "chunk_index": i,  # Ensure chunk_index is always present
                    "chunk_text": chunk,
                    "file_type": doc.get('metadata', {}).get('file_type', 'unknown'),
                    "document_type": doc.get('metadata', {}).get('document_type', 'unknown'),
                    "created_at": datetime.now().isoformat(),
                    "chunk_length": len(chunk),
                    **self._extract_metadata(doc)
                }

                batch.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=metadata
                ))
                total_chunks += 1

                if len(batch) >= 10:
                    self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)
                    logger.info(f"Uploaded 10 chunks from {doc.get('filename', 'unknown')}")
                    batch.clear()
                    gc.collect()

            if batch:
                self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)
                logger.info(f"Uploaded remaining {len(batch)} chunks from {doc.get('filename', 'unknown')}")
                del batch
                gc.collect()

            time.sleep(0.5)

        logger.info(f"Embedded total of {total_chunks} chunks.")
        return total_chunks

    def _extract_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional metadata depending on the file type.
        """
        m = doc.get('metadata', {})
        if m.get('file_type') == 'email':
            return {
                'subject': m.get('subject', ''),
                'from_email': m.get('from', ''),
                'to_email': m.get('to', ''),
                'email_date': m.get('date', '')
            }
        elif m.get('file_type') == 'pdf':
            return {
                'pages': m.get('pages', 0),
                'file_size': m.get('file_size', 0)
            }
        return {}

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Retrieve information about the current Qdrant collection.
        """
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Collection info error: {e}")
            return {}

# Run the embedding process
def main():
    embedder = DocumentEmbedder()

    # Optional: Delete and recreate the collection for clean run
    embedder.delete_existing_collection()
    embedder._setup_collection()

    parsed_file_path = "data/unstructured/parsed.jsonl"
    docs = embedder.load_documents_from_file(parsed_file_path)
    count = embedder.embed_and_store_documents(docs)
    print(f"\nEmbedded {count} chunks from {len(docs)} document(s).")

    info = embedder.get_collection_info()
    print("\nCollection Info:")
    [print(f"  {k}: {v}") for k, v in info.items()]

if __name__ == "__main__":
    main()
