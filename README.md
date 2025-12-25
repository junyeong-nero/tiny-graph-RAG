# Mini-Graph-RAG

A naive implementation of Graph-based Retrieval Augmented Generation (RAG) system without using external GraphRAG packages.

## Overview

Mini-Graph-RAG is a lightweight implementation that extracts knowledge graphs from document data (papers, novels, personal statements, etc.) and enables LLM-powered retrieval using the generated graph structure.

## Features

- **Text Chunking**: Split documents with configurable overlap for context preservation
- **Entity Extraction**: Extract entities (PERSON, ORGANIZATION, PLACE, CONCEPT, EVENT) using OpenAI API
- **Relationship Extraction**: Identify relationships between entities with type and description
- **Knowledge Graph**: Build, merge, and store graphs with entity deduplication
- **Graph Traversal**: BFS-based neighbor discovery and subgraph extraction
- **Relevance Ranking**: Score and rank entities/subgraphs for query relevance
- **Response Generation**: Generate answers using retrieved graph context

## Architecture

```
Document Input
    ↓
Text Chunking (with overlap)
    ↓
Entity & Relationship Extraction (OpenAI API)
    ↓
Knowledge Graph Construction (with entity resolution)
    ↓
Graph Storage (JSON)
    ↓
Query Processing
    ↓
Graph-based Retrieval (BFS + ranking)
    ↓
LLM Response Generation (OpenAI API)
```

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .

# Install with dev dependencies (for testing)
uv pip install -e ".[dev]"
```

## Configuration

Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Optional environment variables:

```bash
export OPENAI_MODEL='gpt-4o-mini'  # Default model
export CHUNK_SIZE='1000'           # Characters per chunk
export CHUNK_OVERLAP='200'         # Overlap between chunks
```

## Usage

### CLI

```bash
# Process a document and create knowledge graph
python main.py process document.txt -o graph.json

# Query an existing knowledge graph
python main.py query "What is the relationship between X and Y?" -g graph.json

# Show graph statistics
python main.py stats -g graph.json

# Interactive mode
python main.py interactive document.txt
python main.py interactive -g graph.json  # Use existing graph
```

### Python API

```python
from mini_graph_rag import GraphRAG

# Initialize the system
rag = GraphRAG()

# Process a document
rag.process_document("path/to/your/document.txt")

# Or process raw text
rag.process_text("Your text content here...")

# Query the knowledge graph
response = rag.query("Your question here")
print(response)

# Save the graph for later use
rag.save_graph("knowledge_graph.json")

# Load a previously saved graph
rag.load_graph("knowledge_graph.json")

# Get graph statistics
stats = rag.get_stats()
print(f"Entities: {stats['entities']}, Relationships: {stats['relationships']}")
```

### Example: Korean Novel Analysis

```python
import os
from mini_graph_rag import GraphRAG

# Paths
DOCUMENT_PATH = "data/현진건-운수좋은날.txt"
GRAPH_PATH = "data/현진건-운수좋은날-knowledge-graph.json"

# Initialize the system
rag = GraphRAG()

# Load existing graph or create new one
if os.path.exists(GRAPH_PATH):
    print(f"Loading existing knowledge graph from {GRAPH_PATH}")
    rag.load_graph(GRAPH_PATH)
else:
    print(f"Creating new knowledge graph from {DOCUMENT_PATH}")
    rag.process_document(DOCUMENT_PATH)
    rag.save_graph(GRAPH_PATH)
    print(f"Knowledge graph saved to {GRAPH_PATH}")

# Get graph statistics
stats = rag.get_stats()
print(f"Entities: {stats['entities']}, Relationships: {stats['relationships']}")

# Query the knowledge graph
query = "김첨지에 대해서 알려줘."
print(f"\nQuery: {query}")
response = rag.query(query)
print(f"Response: {response}")
```

## Project Structure

```
mini-graph-RAG/
├── main.py                      # CLI entry point
├── pyproject.toml               # Project configuration
├── mini_graph_rag/              # Main package
│   ├── __init__.py              # GraphRAG main class
│   ├── config.py                # Configuration management
│   ├── chunking/
│   │   └── chunker.py           # Text chunking with overlap
│   ├── extraction/
│   │   ├── extractor.py         # Entity/relationship extraction
│   │   ├── parser.py            # LLM response parsing
│   │   └── prompts.py           # Extraction prompts
│   ├── graph/
│   │   ├── models.py            # Entity, Relationship, KnowledgeGraph
│   │   ├── builder.py           # Graph construction
│   │   └── storage.py           # JSON/pickle storage
│   ├── retrieval/
│   │   ├── retriever.py         # Main retrieval orchestrator
│   │   ├── traversal.py         # BFS graph traversal
│   │   └── ranking.py           # Relevance scoring
│   └── llm/
│       ├── client.py            # OpenAI API wrapper
│       └── prompts.py           # Response generation prompts
└── tests/                       # Test files
    ├── test_chunking.py
    ├── test_extraction.py
    ├── test_graph.py
    └── test_integration.py
```

## Data Models

### Entity

```python
Entity:
  - entity_id: str (UUID)
  - name: str
  - entity_type: str (PERSON | ORGANIZATION | PLACE | CONCEPT | EVENT | OTHER)
  - description: str
  - source_chunks: list[str]
```

### Relationship

```python
Relationship:
  - relationship_id: str (UUID)
  - source_entity_id: str
  - target_entity_id: str
  - relationship_type: str (e.g., WORKS_FOR, LOCATED_IN, KNOWS)
  - description: str
  - source_chunks: list[str]
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_graph.py -v

# Run with coverage
python -m pytest tests/ --cov=mini_graph_rag
```

## How It Works

### 1. Text Chunking
- Documents are split into chunks (default: 1000 chars)
- Chunks overlap (default: 200 chars) to preserve context
- Sentence boundaries are respected when possible

### 2. Entity & Relationship Extraction
- Each chunk is sent to OpenAI API with extraction prompts
- Entities are classified by type (PERSON, ORGANIZATION, etc.)
- Relationships are extracted with source, target, and type

### 3. Knowledge Graph Construction
- Entities are deduplicated by normalized name
- Duplicate entities are merged (descriptions combined)
- Relationships are added with entity ID references

### 4. Retrieval Mechanism
- Query entities are extracted using LLM
- Matching graph entities are found
- BFS traversal expands to neighboring entities
- Subgraph is ranked by relevance to query

### 5. Response Generation
- Retrieved subgraph is formatted as context
- LLM generates response using graph context
- Entities and relationships are cited

## Limitations

This is a naive implementation focused on learning and experimentation:

- **Scalability**: Not optimized for very large documents (> 100k tokens)
- **Entity Resolution**: Simple name-based matching only
- **Graph Storage**: JSON files, no database support
- **Embeddings**: No vector similarity search
- **Caching**: No caching of LLM calls

For production use cases, consider these improvements or use dedicated GraphRAG frameworks.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
