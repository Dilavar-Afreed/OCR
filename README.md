# Agentic OCR - LangGraph + Chroma API

An intelligent Optical Character Recognition (OCR) system powered by AI agents, combining document processing, vector embeddings, and conversational AI for document analysis and question-answering.

**Author:** Dilavar Afreed  
**Version:** 0.1.0

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Docker Setup](#docker-setup)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Features](#features)
---

## Overview

Agentic OCR is a sophisticated document processing system that:

1. **Extracts Text** from PDF documents using advanced OCR models
2. **Vectorizes** content using OpenAI embeddings
3. **Stores** vectors in ChromaDB for semantic search
4. **Answers Questions** using LangGraph agents with tool integration
5. **Provides Analytics** on ingested documents and collections

The system uses a multi-agent architecture powered by LangGraph, where agents can call tools to search the vector database and provide intelligent, context-aware answers about your documents.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (Web UI)                           │
│              HTML/CSS/JavaScript Interface                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  PDF Upload  │  │  Chat/Ask    │  │ Admin Endpoints      │  │
│  │  & Ingestion │  │  Interface   │  │ (Collections, Stats) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└──────┬──────────────────────────┬──────────────────────────┬────┘
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│ OCR Model    │    │ LangGraph Agent      │    │  Vector Store    │
│ (LightOn     │    │  with Tool Calling   │    │  (ChromaDB)      │
│  OCR 2-1B)   │    │                      │    │                  │
└──────────────┘    └──────────┬───────────┘    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  OpenAI Embeddings   │
                    │  & GPT-4o-mini LLM   │
                    └──────────────────────┘
```

---

## Prerequisites

- **Docker & Docker Compose:** Required for running the application
- **OpenAI API Key:** For embeddings and LLM access
- **System:** Linux/macOS/Windows with at least 4GB RAM
- **GPU (Optional):** CUDA-capable GPU for faster OCR processing

---

## Installation

### Using Docker Compose

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd langgraph-chroma-api
   ```

2. **Create `.env` file** (see [Environment Configuration](#environment-configuration)):
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key.

The complete setup and running instructions are in the [Docker Setup](#docker-setup) section below.

---

## Environment Configuration

Create a `.env` file in the project root with the following variables:

```dotenv
# Required: OpenAI API Key for embeddings and LLM
OPENAI_API_KEY=sk-svcacct-xxxxxxxxxxxxx

# Optional: ChromaDB Collection Name (defaults to "OCR_Documents")
CHROMA_COLLECTION_NAME=OCR_Documents

# Docker-only: ChromaDB Connection (set by docker-compose)
# CHROMA_HOST=chroma
# CHROMA_PORT=8000
```

### Environment Variable Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | Your OpenAI API key for embeddings and LLM inference |
| `CHROMA_COLLECTION_NAME` | ❌ No | `OCR_Documents` | Name of the ChromaDB collection for storing vectors |
| `CHROMA_HOST` | ❌ No | `localhost` | ChromaDB server hostname (Docker only) |
| `CHROMA_PORT` | ❌ No | `8001` | ChromaDB server port (Docker only) |

### Obtaining an OpenAI API Key

1. Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key and paste it in your `.env` file
4. **Important:** Never commit your `.env` file to version control

---

## Running the Application

### Start with Docker Compose

```bash
docker-compose up -d
```

The application will be available at `http://localhost:8000`

### Web Interface

Navigate to `http://localhost:8000` to access the interactive web interface featuring:
- PDF upload form
- Real-time chat interface
- Database statistics viewer
- Collection management

---

## API Endpoints

### 1. **POST `/ingest`** - Ingest PDF Documents

Uploads and processes a PDF file: extracts text using OCR, chunks it, creates embeddings, and stores in ChromaDB.

**Parameters:**
- `file` (FormData, required): PDF file to ingest
- `collection_name` (query string, optional): ChromaDB collection name

**Response:**
```json
{
  "pdf_name": "document.pdf",
  "chunks_processed": 45,
  "status": "success"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf" \
  -F "collection_name=MyDocuments"
```

---

### 2. **POST `/ask`** - Ask Question About Documents

Query the ingested documents using natural language. The agent will search the vector database and provide an answer.

**Request Body:**
```json
{
  "question": "What is the main topic discussed?"
}
```

**Response:**
```json
{
  "question": "What is the main topic discussed?",
  "answer": "Based on the ingested documents, the main topic is... [formatted markdown response]"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document"}'
```

---

### 3. **GET `/visualize/collections`** - List All Collections

Returns all ChromaDB collections and their metadata.

**Response:**
```json
{
  "total_collections": 2,
  "collections": [
    {
      "name": "OCR_Documents",
      "metadata": null
    },
    {
      "name": "MyDocuments",
      "metadata": null
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/visualize/collections
```

---

### 4. **GET `/visualize/stats`** - Collection Statistics

Get detailed statistics about a specific collection.

**Query Parameters:**
- `collection_name` (optional): Collection to query (uses default if not specified)

**Response:**
```json
{
  "total_documents": 150,
  "total_chunks": 150,
  "pdfs": {
    "report.pdf": {
      "chunks": 45,
      "total_chars": 125000,
      "sample_chunks": [
        "Sample chunk content...",
        "Another sample..."
      ]
    },
    "guide.pdf": {
      "chunks": 105,
      "total_chars": 280000,
      "sample_chunks": [
        "Guide content sample...",
        "More guide content..."
      ]
    }
  }
}
```

**Example:**
```bash
curl http://localhost:8000/visualize/stats?collection_name=OCR_Documents
```

---

### 5. **GET `/visualize/documents`** - List Documents

List documents in a collection with pagination.

**Query Parameters:**
- `limit` (optional, default: 10): Maximum number of documents to return
- `collection_name` (optional): Collection to query

**Response:**
```json
{
  "total_in_collection": 150,
  "returned": 3,
  "documents": [
    {
      "id": "report.pdf_chunk_0",
      "content": "Document content preview... [truncated at 500 chars]",
      "source": "report.pdf",
      "chunk": 0,
      "full_length": 1245
    },
    {
      "id": "report.pdf_chunk_1",
      "content": "Next chunk content... [truncated at 500 chars]",
      "source": "report.pdf",
      "chunk": 1,
      "full_length": 1180
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/visualize/documents?limit=5&collection_name=OCR_Documents
```

---

### 6. **GET `/`** - Frontend

Serves the interactive web interface for document upload and querying.

---

## Request/Response Examples

### Complete Workflow Example

#### Step 1: Upload a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_report.pdf"
```

**Response:**
```json
{
  "pdf_name": "sample_report.pdf",
  "chunks_processed": 42,
  "status": "success"
}
```

#### Step 2: Check Collection Statistics

```bash
curl http://localhost:8000/visualize/stats
```

**Response:**
```json
{
  "total_documents": 42,
  "total_chunks": 42,
  "pdfs": {
    "sample_report.pdf": {
      "chunks": 42,
      "total_chars": 112000,
      "sample_chunks": [
        "The quarterly report shows a 15% increase...",
        "Financial results for Q3 demonstrate..."
      ]
    }
  }
}
```

#### Step 3: Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue increase mentioned?"}'
```

**Response:**
```json
{
  "question": "What was the revenue increase mentioned?",
  "answer": "## Revenue Increase\n\nBased on the quarterly report, the **revenue increased by 15%** compared to the previous quarter.\n\n### Key Details:\n- **Q3 Financial Results**: Demonstrated strong growth\n- **Overall Performance**: Exceeded projections\n- **Forward Outlook**: Positive trajectory expected\n\nThis represents one of the strongest quarters on record."
}
```

---

## Docker Setup

### Docker Compose Architecture

The `docker-compose.yml` orchestrates two main services:

1. **chromadb/chroma** (Port 8001)
   - Vector database for storing embeddings
   - Persistent storage with volume

2. **ocr_app** (Port 8000)
   - FastAPI application
   - Depends on chroma service
   - Automatically restarts unless stopped


### Quick Start with Docker

1. **Prepare environment:**
   ```bash
   # Create .env file with OpenAI API key
   echo "OPENAI_API_KEY=your-key-here" > .env
   echo "CHROMA_COLLECTION_NAME=OCR_Documents" >> .env
   ```

2. **Start containers:**
   ```bash
   docker-compose up -d
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Access the application:**
   - API: http://localhost:8000
   - ChromaDB: http://localhost:8001
   - Frontend: http://localhost:8000

5. **Stop containers:**
   ```bash
   docker-compose down
   ```

6. **Remove everything including volumes:**
   ```bash
   docker-compose down -v
   ```

## Project Structure

```
langgraph-chroma-api/
├── app/                          # Main application package
│   ├── main.py                   # FastAPI application & endpoints
│   ├── agent.py                  # LangGraph agent configuration
│   ├── tools.py                  # Tool definitions for agent
│   ├── vector_store.py           # ChromaDB operations
│   └── static/                   # Frontend assets
│       ├── index.html            # Web interface
│       ├── script.js             # Client-side logic
│       └── style.css             # Styling
│
├── docker-compose.yml            # Container orchestration
├── Dockerfile                    # Application container definition
├── pyproject.toml                # Python dependencies & metadata
├── run.py                        # Server startup script
├── model_inference.py            # Model inference utilities
├── README.md                     # This file
└── .env                          # Environment variables (git-ignored)
```

---

## Technology Stack

### Core Framework
- **FastAPI** (0.120+) - Modern, fast web framework
- **Uvicorn** (0.38+) - ASGI server

### AI & Machine Learning
- **OpenAI** (2.6+) - API client for embeddings & GPT-4o-mini
- **LangChain** (1.2+) - LLM framework
- **LangChain OpenAI** (1.0+) - OpenAI integration
- **LangGraph** (1.0+) - Agent orchestration framework
- **Transformers** (4.40+) - HuggingFace models (OCR)
- **PyTorch** (2.0+) - Deep learning framework
- **TensorFlow/Torch** - GPU support (CUDA/MPS)

### Document Processing
- **PyMuPDF** (1.24+) - PDF text extraction & rendering
- **PyPDF** (4.0+) - PDF manipulation
- **Pillow** (10.0+) - Image processing

### Data & Storage
- **ChromaDB** (0.4+) - Vector database
- **LangChain Text Splitters** (1.0+) - Document chunking
- **TikToken** (0.12+) - Token counting

### Utilities
- **python-dotenv** (1.2+) - Environment variable management
- **python-multipart** (0.0+) - Form data parsing

---

## Features

### Document Processing
- ✅ **Advanced OCR**: LightOnOCR-2-1B model for high-quality text extraction
- ✅ **Multi-page PDF Support**: Processes all pages with page markers
- ✅ **Intelligent Chunking**: Recursive text splitting with overlap for context preservation
- ✅ **GPU Acceleration**: Automatic device detection (CUDA/MPS/CPU)

### Vector Database
- ✅ **Semantic Search**: OpenAI embeddings for meaning-based document search
- ✅ **Multiple Collections**: Organize documents by project/topic
- ✅ **Persistent Storage**: Data survives container restarts
- ✅ **Metadata Tracking**: Track document source and chunk information

### AI Agent System
- ✅ **Tool-Calling Agents**: LangGraph-based agents with tool integration
- ✅ **Intelligent Responses**: GPT-4o-mini with markdown formatting
- ✅ **Context-Aware Answers**: Vector search-grounded responses
- ✅ **Error Handling**: Graceful error messages with detailed feedback

### Analytics & Monitoring
- ✅ **Collection Statistics**: Document count, chunk distribution
- ✅ **Document Listing**: Preview documents with metadata
- ✅ **Collection Management**: View all active collections
- ✅ **Source Tracking**: Know which PDFs contributed to answers

### User Interface
- ✅ **Web-Based Interface**: Interactive frontend with no installation needed
- ✅ **Real-time Chat**: Instant question answering
- ✅ **File Upload**: Drag-and-drop PDF upload
- ✅ **Statistics Dashboard**: Monitor your collections

---

## License

Created by Dilavar Afreed - Agentic OCR v0.1.0

---

## Support & Contribution

For issues, questions, or contributions, please refer to the project repository.
