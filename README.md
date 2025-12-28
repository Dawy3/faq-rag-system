# FAQ RAG System 🚀

A modular, production-grade Retrieval-Augmented Generation (RAG) system designed for FAQ automation. Built with **FastAPI**, **LangGraph** (for the decision brain), and **Pinecone** (vector storage).



## ✨ Features
- **Intelligent Query Rewriting**: Optimizes user questions for better vector search.
- **Strict Relevance Grading**: Prevents hallucinations by filtering out irrelevant context.
- **Production Structure**: Modular directory layout for scalability.
- **CI/CD Ready**: Integrated with GitHub Actions for automated testing and Docker builds.
- **API Documentation**: Automatic Swagger UI via FastAPI.

---

## 🛠️ Tech Stack
- **Framework**: FastAPI
- **Orchestration**: LangGraph
- **LLM**: OpenRouter (GPT-3.5/4)
- **Vector DB**: Pinecone
- **Embeddings**: HuggingFace (MiniLM)
- **DevOps**: Docker, GitHub Actions

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone [https://github.com/Dawy3/faq-rag-system.git](https://github.com/Dawy3/faq-rag-system.git)
cd faq-rag-system

```

### 2. Environment Setup

Create a `.env` file in the root:

```env
OPENROUTER_API_KEY=your_key
PINECONE_API_KEY=your_key
MODEL_NAME=openai/gpt-3.5-turbo

```

### 3. Run with Docker (Recommended)

```bash
docker build -t faq-app .
docker run -p 8000:8000 --env-file .env faq-app

```

### 4. Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

```

---

## 🧪 Testing

Run the automated test suite to ensure the RAG logic and API endpoints are working correctly:

```bash
pytest tests/

```

---

## 📡 API Endpoints

* `POST /api/v1/document/upload`: Upload a PDF to index it into the FAQ knowledge base.
* `POST /api/v1/query`: Ask a question based on the uploaded documents.
* `GET /docs`: Interactive Swagger documentation.
