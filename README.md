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

Mini-Graph-RAG can be configured using a `config.yaml` file or environment variables. Environment variables take precedence over settings defined in the YAML file.

### YAML Configuration

Create a `config.yaml` in your project root or current working directory:

```yaml
# OpenAI API Settings
openai:
    # Base URL for OpenAI-compatible API (optional)
    # Examples: http://localhost:11434/v1 (Ollama), Azure OpenAI endpoints, etc.
    base_url: null
    # Model name to use
    model: "gpt-4o-mini"
    # Hyperparameters
    temperature: 0.0
    max_tokens: 4096

# Text Chunking Settings
chunking:
    chunk_size: 1000
    chunk_overlap: 200
```

### Environment Variables

The OpenAI API key is required and must be set via environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Optional variables (overrides `config.yaml`):

| Variable | YAML Path | Description | Default |
|----------|-----------|-------------|---------|
| `OPENAI_MODEL` | `openai.model` | Model name to use | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | `openai.base_url` | Base URL for API | `null` |
| `OPENAI_TEMPERATURE` | `openai.temperature` | LLM temperature | `0.0` |
| `OPENAI_MAX_TOKENS` | `openai.max_tokens` | Max tokens for response | `4096` |
| `CHUNK_SIZE` | `chunking.chunk_size` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | `chunking.chunk_overlap` | Overlap between chunks | `200` |

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

**Output:**

```text
Query: 김첨지에 대해서 알려줘.
Response: 요약 — 김첨지에 대해 알려드리겠습니다.

- 신분·역할: 이야기의 주인공이자 인력거꾼입니다. (엔티티: 김첨지 — 이야기 속 주인공, 인력거꾼)
- 현재 상황·행동:
  - 술집에 머물며 술을 마십니다. (김첨지 --[LOCATED_IN]--> 술집, 김첨지 --[DRINKS]--> 술)
  - 자기 아내의 시체를 집에 뻐들쳐 놓았다고 진술합니다. (김첨지 --[PLACED]--> 마누라 시체)
  - 그날 전차 정류장에 갔었고, 전차 정류장 근처를 빙빙 돌며 손님을 기다리려는 계획을 세웠습니다. (김첨지 --[WENT_TO]--> 전차 정류장, 김첨지 --[PLANS_TO_WAIT_AT]--> 전차 정류장)
  - 학생 승객을 정거장까지 태워다 주었습니다(그 학생을 데려다 주었다). (김첨지 --[TRANSPORTS_TO]--> 그 학생)
  - 자기를 불러 멈춘 사람이 학교 학생임을 알아보았습니다. (김첨지 --[RECOGNIZED]--> 그 학교 학생; 사람 --[ALIAS_OF]--> 그 학교 학생)
- 성격·내면·기억:
  - 약을 쓰면 병이 재미를 붙여 자꾸 온다는 신조(신념)를 가지고 있습니다. (김첨지 --[HOLDS_BELIEF]--> 신조(信條))
  - 기적에 가까운 벌이를 했다는 기쁨을 오래 간직하려 합니다. (김첨지 --[REMINISCES_ABOUT]--> 기적에 가까운 벌이)
  - 술을 마시며 감정을 드러내고(울고 웃는 등) 아내의 죽음을 호소하는 장면이 있습니다. (엔티티 설명: 술집에 있는 주정뱅이 남성, 이야기하고 술을 마시며 아내의 죽음을 호소함)
- 외형·묘사:
  - 바짝 마른 얼굴과 턱밑에 특이한 수염이 있는 것으로 묘사됩니다. (김첨지 --[HAS_ATTRIBUTE]--> 김첨지)
  - 걸음걸이가 스케이트 타는 모양으로 비유되어 묘사됩니다. (김첨지 --[RELATED_TO]--> 스케이트)

사용한 엔티티·관계 근거:
- 엔티티: 김첨지 (이야기 속 주인공, 인력거꾼)
- 관계들:
  - 김첨지 --[LOCATED_IN]--> 술집
  - 김첨지 --[DRINKS]--> 술
  - 김첨지 --[PLACED]--> 마누라 시체
  - 김첨지 --[WENT_TO]--> 전차 정류장
  - 김첨지 --[PLANS_TO_WAIT_AT]--> 전차 정류장
  - 김첨지 --[TRANSPORTS_TO]--> 그 학생
  - 김첨지 --[RECOGNIZED]--> 그 학교 학생
  - 사람 --[ALIAS_OF]--> 그 학교 학생
  - 김첨지 --[HOLDS_BELIEF]--> 신조(信條)
  - 김첨지 --[REMINISCES_ABOUT]--> 기적에 가까운 벌이
  - 김첨지 --[RELATED_TO]--> 스케이트
  - 김첨지 --[HAS_ATTRIBUTE]--> 김첨지
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
