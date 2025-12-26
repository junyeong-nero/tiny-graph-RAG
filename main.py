"""CLI entry point for Mini-Graph-RAG."""

import argparse
import sys

from mini_graph_rag import GraphRAG


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Mini-Graph-RAG: Graph-based Retrieval Augmented Generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("document", help="Path to the document to process")
    process_parser.add_argument(
        "-o", "--output", default="graph.json", help="Output path for the graph (default: graph.json)"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "-g", "--graph", default="graph.json", help="Path to the graph file (default: graph.json)"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show graph statistics")
    stats_parser.add_argument(
        "-g", "--graph", default="graph.json", help="Path to the graph file (default: graph.json)"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive mode with a document"
    )
    interactive_parser.add_argument(
        "document", nargs="?", help="Path to the document (optional if graph exists)"
    )
    interactive_parser.add_argument(
        "-g", "--graph", help="Path to existing graph file"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize knowledge graph"
    )
    visualize_parser.add_argument(
        "-g", "--graph", required=True, help="Path to graph JSON file"
    )
    visualize_parser.add_argument(
        "-o", "--output", default="graph_viz.html", help="Output HTML file (default: graph_viz.html)"
    )
    visualize_parser.add_argument(
        "--filter-type", nargs="+", help="Filter by entity types (e.g., PERSON PLACE)"
    )
    visualize_parser.add_argument(
        "--min-weight", type=float, default=0.0, help="Minimum relationship weight (default: 0.0)"
    )
    visualize_parser.add_argument(
        "--max-nodes", type=int, default=200, help="Maximum nodes to display (default: 200)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "process":
            run_process(args)
        elif args.command == "query":
            run_query(args)
        elif args.command == "stats":
            run_stats(args)
        elif args.command == "interactive":
            run_interactive(args)
        elif args.command == "visualize":
            run_visualize(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_process(args):
    """Process a document and save the graph."""
    print(f"Processing document: {args.document}")

    rag = GraphRAG()
    rag.process_document(args.document)
    rag.save_graph(args.output)

    stats = rag.get_stats()
    print(f"\nGraph statistics:")
    print(f"  Entities: {stats['entities']}")
    print(f"  Relationships: {stats['relationships']}")
    if stats.get("entity_types"):
        print(f"  Entity types: {stats['entity_types']}")


def run_query(args):
    """Query an existing graph."""
    rag = GraphRAG()
    rag.load_graph(args.graph)

    print(f"\nQuestion: {args.question}")
    print("-" * 50)

    response = rag.query(args.question)
    print(response)


def run_stats(args):
    """Show statistics for a graph."""
    rag = GraphRAG()
    rag.load_graph(args.graph)

    stats = rag.get_stats()
    print("Graph Statistics:")
    print(f"  Total entities: {stats['entities']}")
    print(f"  Total relationships: {stats['relationships']}")

    if stats.get("entity_types"):
        print("\n  Entity types:")
        for etype, count in sorted(stats["entity_types"].items()):
            print(f"    {etype}: {count}")

    if stats.get("relationship_types"):
        print("\n  Relationship types:")
        for rtype, count in sorted(stats["relationship_types"].items()):
            print(f"    {rtype}: {count}")


def run_interactive(args):
    """Run interactive mode."""
    rag = GraphRAG()

    # Load or process
    if args.graph:
        rag.load_graph(args.graph)
    elif args.document:
        rag.process_document(args.document)
    else:
        print("Error: Provide either a document or a graph file.", file=sys.stderr)
        sys.exit(1)

    print("\nInteractive mode. Type 'quit' or 'exit' to stop.")
    print("-" * 50)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = rag.query(question)
        print(f"\nAnswer: {response}")


def run_visualize(args):
    """Visualize a knowledge graph."""
    from mini_graph_rag.visualization import PyVisVisualizer
    from mini_graph_rag.graph.storage import GraphStorage

    print(f"Loading graph from: {args.graph}")

    # Load graph
    storage = GraphStorage()
    graph = storage.load_json(args.graph)

    print(f"Graph loaded: {len(graph.entities)} entities, {len(graph.relationships)} relationships")

    # Create visualizer
    viz = PyVisVisualizer(
        graph=graph,
        filter_types=args.filter_type,
        min_weight=args.min_weight,
        max_nodes=args.max_nodes,
    )

    # Generate and save
    print("Generating visualization...")
    viz.generate()
    viz.save(args.output)

    # Open in browser
    viz.show()


if __name__ == "__main__":
    main()
