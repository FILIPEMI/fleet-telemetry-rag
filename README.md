# Fleet Telemetry RAG System

A RAG (Retrieval-Augmented Generation) system that connects to fleet telemetry APIs and lets you ask natural language questions about your vehicle data. Built this to make fleet data more accessible through conversational AI instead of complex dashboard queries.

## What it does

The system fetches telemetry data from fleet management APIs, stores it in a vector database, and uses AI to answer questions about vehicle performance, locations, and operational insights. Instead of writing complex SQL queries or navigating dashboards, just ask questions like "which vehicles need maintenance?" or "what's the fuel efficiency trend this month?"

## Tech Stack

- **Python 3.8+** - Core language
- **LangChain** - RAG pipeline and document processing
- **HuggingFace** - Embeddings and language models (Mistral-7B)
- **ChromaDB** - Vector database for semantic search
- **Pandas** - Data processing and CSV handling
- **Requests** - API client for telemetry data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment variables:
```bash
cp .env.example .env
# Edit .env with your actual API credentials
```

3. Run the application:
```bash
python fleet_telemetry_rag.py
```

## Configuration

Set these environment variables in your `.env` file:

- `FLEET_API_ENDPOINT` - Your fleet telemetry API endpoint
- `FLEET_BEARER_TOKEN` - API authentication token
- `HUGGINGFACE_API_TOKEN` - HuggingFace API token for LLM access

## How it works

1. **Data Fetching** - Connects to your fleet API and pulls current telemetry data
2. **Processing** - Converts JSON responses to CSV and splits into chunks
3. **Vector Storage** - Creates embeddings and stores in ChromaDB for fast retrieval
4. **RAG Pipeline** - Uses retriever + Mistral-7B to answer questions with context

The vector database persists in `docs/chroma_fleet/` so you don't need to rebuild it every time.

## Challenges I solved

- **API Rate Limits** - Added proper error handling and response validation
- **Memory Management** - Chunking large datasets to prevent embedding model overload
- **Token Security** - Moved all credentials to environment variables
- **Data Consistency** - Handles both single record and batch API responses

## What I learned

Working with vector databases was new to me - figuring out optimal chunk sizes and overlap settings took some experimentation. Also learned how different embedding models affect retrieval quality, and why semantic search works better than keyword matching for telemetry data.

The LangChain ecosystem moves fast, had to update from deprecated RetrievalQA to the newer create_retrieval_chain approach mid-development.

## Future improvements

- Add streaming responses for longer queries
- Implement caching layer for frequently asked questions
- Support multiple fleet data sources
- Add basic authentication and user management
- Create a simple web interface instead of CLI

## Sample queries

- "What vehicles are currently active?"
- "Show me fuel efficiency trends"
- "Which vehicles need maintenance soon?"
- "What's the average speed across the fleet today?"

Built this originally to help operations teams get quick insights without learning complex query languages or waiting for dashboard loads.