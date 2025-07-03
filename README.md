# RAGxplore
RAGxplore: AI-Powered Enterprise Search using Retrieval-Augmented Generation
# RAGxplore

**RAGxplore** is an AI-powered enterprise discovery and intelligence engine that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-rich answers to complex business queries. By combining structured data (SQL), unstructured content (PDFs, emails), and graph-based knowledge (Neo4j), RAGxplore helps businesses retrieve the right information from diverse sources — intelligently, quickly, and transparently.

---

## Features

- RAG-Driven Search: Combines semantic search with structured SQL and graph-based retrieval.
- Multi-Source Integration: Works with CSV files, PDFs, emails, text files, and knowledge graphs.
- Zero-Shot Reasoning: Uses OpenAI's `gpt-3.5-turbo` to generate answers using retrieved context.
- Streamlit UI: Intuitive dashboard for querying, exploring citations, and giving feedback.
- Feedback Loop: Real-time answer rating system to fine-tune the model and improve accuracy.
- PII Detection & Guardrails: Flags sensitive content to ensure compliance and safety.

---

## Architecture Overview

**RAGxplore** is built using a modular and extensible architecture powered by LangChain and custom retrieval tools:

### 1. Document Ingestion
- Structured data loaded from CSVs into DuckDB
- Unstructured documents (PDF, TXT, EML) parsed and saved in `.jsonl`
- Metadata added for traceability

### 2. Embedding & Storage
- Text split using `RecursiveCharacterTextSplitter`
- Embeddings generated with `all-MiniLM-L6-v2` from Hugging Face
- Stored in Qdrant (vector database) for semantic retrieval

### 3. Retrieval Tools
- SQLRetriever: Converts natural language to SQL (via LangChain) to query DuckDB
- VectorRetriever: Retrieves top-K similar chunks from Qdrant
- GraphRetriever: Queries relationships from Neo4j using Cypher

### 4. Multi-Tool Agent (LangChain AgentExecutor)
- Dynamically chooses the right retrieval tool based on user query
- Supports tool chaining (one tool’s output becomes another’s input)
- Uses `gpt-3.5-turbo` to synthesize answers from retrieved content

### 5. UI & Feedback
- Streamlit App allows:
  - Query input
  - Domain filter (Finance, Biotech, etc.)
  - Confidence, date, and source filters
  - Feedback buttons (Helpful / Not Helpful)
- Observability Dashboard shows:
  - Query trends
  - Feedback stats
  - Average response times

---

## Folder Structure


RAGxplore/
├── data/ # Structured & unstructured input files
│ ├── structured/
│ └── unstructured/
├── src/
│ ├── ingest/ # Document parsing & metadata tagging
│ ├── retrievers/ # SQL, vector, and graph retrieval logic
│ └── rag_tool.py # Core RAG logic and LangChain agent
├── dashboards/ # Streamlit visualizations for observability
├── logs/ # Feedback logs
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
└── README.md # This file


---

## Tech Stack

| Component           | Technology                     |
|---------------------|--------------------------------|
| Language Model      | OpenAI GPT-3.5-turbo           |
| Embeddings          | all-MiniLM-L6-v2 (Hugging Face)|
| Semantic Search     | Qdrant                         |
| Structured Querying | DuckDB + LangChain SQL chain   |
| Graph Retrieval     | Neo4j + Cypher                 |
| Framework           | LangChain + Streamlit          |
| Feedback Logging    | JSONL + Streamlit              |
| Visualization       | Plotly                         |

---

## Future Enhancements

- LangGraph Integration for smarter graph reasoning
- Fine-Tuning with Feedback Data using LoRA + PEFT
- Cloud-Native Architecture for scalability and deployment
- Advanced Query Routing based on query type (semantic, SQL, graph)

---

## Developed By

**Moulika Challa**  
Master’s in Computer Science – California State University, Sacramento  
Data Science Intern | OpenAI Developer | Machine Learning & RAG Enthusiast

---

## Contact

For questions or collaboration opportunities:  
Email: moulikachalla@gmail.com  
LinkedIn: [Moulika Challa](https://www.linkedin.com/in/moulika-challa)

---

## License

This project is licensed under the **MIT License** – see `LICENSE` file for details.

