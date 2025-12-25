"""Prompts for response generation."""

RESPONSE_GENERATION_SYSTEM = """You are a helpful assistant that answers questions based on the provided knowledge graph context.
Use only the information provided in the context to answer.
If the context doesn't contain enough information, say so clearly.
Cite the entities and relationships you use in your answer."""


def build_response_prompt(query: str, context: str) -> str:
    """Build prompt for response generation.

    Args:
        query: User's question
        context: Formatted context from knowledge graph

    Returns:
        Formatted prompt for the LLM
    """
    return f"""Context from Knowledge Graph:
{context}

User Question: {query}

Please answer the question based on the context above."""
