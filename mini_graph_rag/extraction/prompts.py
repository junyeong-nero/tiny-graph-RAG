"""Prompts for entity and relationship extraction."""

EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting entities and relationships from text.
Extract all named entities and the relationships between them.

Entity types to identify:
- PERSON: Individual people, characters
- ORGANIZATION: Companies, institutions, groups
- PLACE: Locations, cities, countries
- CONCEPT: Abstract ideas, theories, technologies
- EVENT: Historical events, meetings, incidents
- OTHER: Any other notable entities

For relationships, use verb-based types like:
- WORKS_FOR, LOCATED_IN, CREATED_BY, KNOWS, PART_OF, RELATED_TO, etc.

Return your response as a JSON object with this exact structure:
{
    "entities": [
        {"name": "Entity Name", "type": "ENTITY_TYPE", "description": "Brief description"}
    ],
    "relationships": [
        {"source": "Source Entity Name", "target": "Target Entity Name", "type": "RELATIONSHIP_TYPE", "description": "Brief description of the relationship"}
    ]
}

Important:
- Extract ALL entities mentioned in the text
- Entity names in relationships must exactly match entity names in the entities list
- Be thorough but precise - only extract what is explicitly stated or clearly implied"""


def build_extraction_prompt(text: str) -> str:
    """Build the user prompt for extraction.

    Args:
        text: The text to extract from

    Returns:
        Formatted prompt
    """
    return f"""Extract entities and relationships from the following text:

---
{text}
---

Return the entities and relationships as JSON."""
