"""Apply Entity Resolution to all *-KG.json files."""

from pathlib import Path

from tiny_graph_rag.config import Config
from tiny_graph_rag.graph import GraphStorage, LLMEntityResolver
from tiny_graph_rag.llm import OpenAIClient


def main() -> None:
    config = Config.from_env()
    llm_client = OpenAIClient(api_key=config.openai_api_key, model=config.model_name)
    resolver = LLMEntityResolver(llm_client=llm_client)
    storage = GraphStorage()

    kg_dir = Path(config.kg_dir or "data/kg")
    kg_files = sorted(kg_dir.glob("*-KG.json"))

    if not kg_files:
        print("No *-KG.json files found.")
        return

    for kg_path in kg_files:
        print(f"\n{'='*60}")
        print(f"Processing: {kg_path.name}")
        print(f"{'='*60}")

        graph = storage.load_json(kg_path)
        before_entities = len(graph.entities)
        before_rels = len(graph.relationships)

        print(f"  Before ER: {before_entities} entities, {before_rels} relationships")

        resolver.resolve(graph)

        after_entities = len(graph.entities)
        after_rels = len(graph.relationships)
        merged = before_entities - after_entities

        print(f"  After ER:  {after_entities} entities, {after_rels} relationships")
        print(f"  Merged:    {merged} entities")

        storage.save_json(graph, kg_path)
        print(f"  Saved:     {kg_path}")


if __name__ == "__main__":
    main()
