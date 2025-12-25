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
