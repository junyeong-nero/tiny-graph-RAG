"""CLI entry point for Tiny-Graph-RAG."""

import argparse
import sys

from tiny_graph_rag import GraphRAG


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Tiny-Graph-RAG: Graph-based Retrieval Augmented Generation"
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

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate retrieval quality against a dataset"
    )
    eval_parser.add_argument(
        "--dataset", required=True, help="Path to evaluation dataset (JSONL)"
    )
    eval_parser.add_argument(
        "-g", "--graph", required=True, help="Path to graph JSON file"
    )
    eval_parser.add_argument(
        "--top-k", type=int, default=5, help="Top-k entities for metrics (default: 5)"
    )
    eval_parser.add_argument(
        "--hops", type=int, default=2, help="BFS traversal depth (default: 2)"
    )
    eval_parser.add_argument(
        "-o", "--output", default="eval_results.json",
        help="Output JSON file for results (default: eval_results.json)"
    )
    eval_parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip LLM response generation (retrieval-only evaluation)",
    )
    eval_parser.add_argument(
        "--price-per-1k-input", type=float, default=0.00015,
        help="USD per 1000 input tokens (default: 0.00015)",
    )
    eval_parser.add_argument(
        "--price-per-1k-output", type=float, default=0.0006,
        help="USD per 1000 output tokens (default: 0.0006)",
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
        elif args.command == "eval":
            run_eval(args)
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
    print("\nGraph statistics:")
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
    from tiny_graph_rag.visualization import PyVisVisualizer
    from tiny_graph_rag.graph.storage import GraphStorage

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


def run_eval(args):
    """Run retrieval evaluation against a dataset."""
    from tiny_graph_rag.evaluation.runner import EvaluationRunner, save_eval_output
    from tiny_graph_rag.graph.storage import GraphStorage
    from tiny_graph_rag.llm import OpenAIClient
    from tiny_graph_rag.config import Config

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.hops < 0:
        raise ValueError("--hops must be >= 0")
    if args.price_per_1k_input < 0:
        raise ValueError("--price-per-1k-input must be >= 0")
    if args.price_per_1k_output < 0:
        raise ValueError("--price-per-1k-output must be >= 0")

    print(f"Loading graph from: {args.graph}")
    storage = GraphStorage()
    graph = storage.load_json(args.graph)
    print(f"Graph loaded: {len(graph.entities)} entities, {len(graph.relationships)} relationships")

    config = Config.from_env()
    llm_client = OpenAIClient(
        api_key=config.openai_api_key,
        model=config.model_name,
        base_url=config.base_url,
    )

    runner = EvaluationRunner(
        graph=graph,
        llm_client=llm_client,
        top_k=args.top_k,
        hops=args.hops,
        price_per_1k_input=args.price_per_1k_input,
        price_per_1k_output=args.price_per_1k_output,
        skip_generation=args.skip_generation,
    )

    print(f"Running evaluation on: {args.dataset}")
    print(f"  top_k={args.top_k}, hops={args.hops}")
    results, summary = runner.run(args.dataset)

    save_eval_output(results, summary, args.output)
    print(f"\nResults saved to: {args.output}")

    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({summary.num_examples} examples, k={summary.k})")
    print(f"{'='*50}")
    print(f"  Avg Precision@{summary.k}: {summary.avg_precision_at_k:.4f}")
    print(f"  Avg Recall@{summary.k}:    {summary.avg_recall_at_k:.4f}")
    print(f"  Avg MRR:            {summary.avg_mrr:.4f}")
    print(f"  Avg nDCG@{summary.k}:      {summary.avg_ndcg_at_k:.4f}")
    print(f"  Avg Latency:        {summary.avg_latency_seconds:.4f}s")
    print(f"  Total Tokens:       {summary.total_tokens}")
    print(f"    Prompt:           {summary.total_prompt_tokens}")
    print(f"    Completion:       {summary.total_completion_tokens}")
    print(f"  Estimated Cost:     ${summary.total_estimated_cost_usd:.6f}")


if __name__ == "__main__":
    main()
