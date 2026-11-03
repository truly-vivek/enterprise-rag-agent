# Enterprise RAG Agent

A document-grounded question-answering API built with FastAPI, OpenAI models, ChromaDB, and the Model Context Protocol (MCP).

Upload PDF, Markdown, or plain-text documents, index their contents, and ask questions against the resulting knowledge base. Answers include the retrieved source passages used to generate them.

## Features

- PDF, Markdown, and plain-text ingestion
- Configurable text chunking with overlap
- OpenAI embeddings and chat completion
- Persistent ChromaDB vector storage
- Semantic retrieval with configurable result count
- Answers grounded in retrieved document excerpts
- Source references returned with each answer
- FastAPI endpoints and interactive API documentation
- MCP tool for searching the knowledge base
- Docker support

## Requirements

- Python 3.11+
- OpenAI API key
- Docker (optional)

## Run locally

Clone the repository:

```bash
git clone https://github.com/truly-vivek/enterprise-rag-agent.git
cd enterprise-rag-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Keep the real `.env` file private and out of version control.

Start the API:

```bash
uvicorn app.main:app --reload
```

The service runs at http://localhost:8000. Interactive API documentation is available at http://localhost:8000/docs.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health |
| POST | `/documents` | Upload and index a PDF, TXT, or MD file |
| GET | `/documents` | List indexed documents and chunk counts |
| POST | `/query` | Ask a question against indexed documents |

### Upload a document

```bash
curl -X POST "http://localhost:8000/documents" \
  -F "file=@employee-handbook.pdf"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the annual leave policy?","top_k":5}'
```

The response includes an answer and a list of source records. Each source identifies the document, chunk ID, retrieved text, and vector distance.

## MCP server

The project exposes a `search_knowledge_base` MCP tool. Run it using:

```bash
python -m app.mcp_server
```

Configure an MCP-compatible client to launch this command from the repository directory. The server uses the same environment variables and persistent vector store as the API.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key for embeddings and chat |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | Vector database location |
| `CHROMA_COLLECTION` | `knowledge_base` | Collection name |
| `MAX_UPLOAD_MB` | `20` | Maximum upload size |
| `CHUNK_SIZE` | `1000` | Chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `5` | Default retrieval count |

## Docker

Create a local `.env` file, then run:

```bash
docker compose up --build
```

The `data` directory is mounted to persist the vector database between restarts.

## Tests

Install the dependencies and run:

```bash
pytest
```

## Production considerations

This repository provides a working development foundation. Before handling confidential business data or exposing the service publicly, add authentication, authorisation, tenant isolation, upload scanning, rate limiting, audit logging, secrets management, retention policies, and TLS. Review the data handling terms of the model and hosting providers before processing sensitive information.

## License

See [LICENSE](LICENSE).
