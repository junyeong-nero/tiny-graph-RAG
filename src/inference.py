import os
from tiny_graph_rag import GraphRAG

# Paths
DOCUMENT_PATH = "data/novels/김유정-동백꽃.txt"
GRAPH_PATH = "data/kg/김유정-동백꽃-KG.json"

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
query = "이 소설의 주인공에 대해서 설명해줘."
print(f"\nQuery: {query}")
response = rag.query(query)
print(f"Response: {response}")
