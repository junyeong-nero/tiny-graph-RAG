"""Evaluation runner with latency, token, and cost tracking."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..graph.models import KnowledgeGraph
from ..llm.client import OpenAIClient
from ..llm.prompts import RESPONSE_GENERATION_SYSTEM, build_response_prompt
from ..retrieval.ranking import SubgraphRanker
from ..retrieval.retriever import GraphRetriever
from .dataset import EvalExample, load_dataset
from .metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)


@dataclass
class TokenUsage:
    """Tracks token usage for a single evaluation example."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: "TokenUsage") -> None:
        """Accumulate usage counts."""
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens


@dataclass
class EvalResult:
    """Result for a single evaluation example."""

    example_id: str
    query: str
    reference_entities: list[str]
    retrieved_entities: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_seconds: float
    token_usage: TokenUsage
    estimated_cost_usd: float
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "example_id": self.example_id,
            "query": self.query,
            "reference_entities": self.reference_entities,
            "retrieved_entities": self.retrieved_entities,
            "metrics": {
                "precision_at_k": self.precision_at_k,
                "recall_at_k": self.recall_at_k,
                "mrr": self.mrr,
                "ndcg_at_k": self.ndcg_at_k,
            },
            "latency_seconds": self.latency_seconds,
            "token_usage": {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
            },
            "estimated_cost_usd": self.estimated_cost_usd,
            "tags": self.tags,
        }


@dataclass
class EvalSummary:
    """Aggregated evaluation summary."""

    num_examples: int
    k: int
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_mrr: float
    avg_ndcg_at_k: float
    avg_latency_seconds: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_estimated_cost_usd: float

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "num_examples": self.num_examples,
            "k": self.k,
            "avg_precision_at_k": self.avg_precision_at_k,
            "avg_recall_at_k": self.avg_recall_at_k,
            "avg_mrr": self.avg_mrr,
            "avg_ndcg_at_k": self.avg_ndcg_at_k,
            "avg_latency_seconds": self.avg_latency_seconds,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
        }


class _UsageTrackingClient:
    """Wraps OpenAIClient to capture token usage from responses.

    Monkey-patches the underlying client.chat.completions.create to
    intercept the response.usage field without changing the public API.
    """

    def __init__(self, llm_client: OpenAIClient):
        self._llm_client = llm_client
        self._accum_usage: TokenUsage = TokenUsage()
        self._original_create = llm_client.client.chat.completions.create
        # Wrap the create method to capture usage
        llm_client.client.chat.completions.create = self._tracked_create  # type: ignore[method-assign]

    def _tracked_create(self, **kwargs):  # type: ignore[no-untyped-def]
        """Intercept completions.create to capture usage."""
        response = self._original_create(**kwargs)
        if hasattr(response, "usage") and response.usage is not None:
            self._accum_usage.add(TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            ))
        return response

    def get_and_reset_usage(self) -> TokenUsage:
        """Return accumulated usage and reset to zero."""
        usage = self._accum_usage
        self._accum_usage = TokenUsage()
        return usage

    def restore(self) -> None:
        """Restore the original create method."""
        self._llm_client.client.chat.completions.create = self._original_create  # type: ignore[method-assign]


class EvaluationRunner:
    """Runs retrieval evaluation over a dataset.

    Args:
        graph: The KnowledgeGraph to evaluate against.
        llm_client: OpenAI client used by the retriever.
        top_k: Number of top entities for retrieval metrics (default: 5).
        hops: BFS hop depth for graph traversal (default: 2).
        price_per_1k_input: USD per 1000 input tokens (default: 0.15e-3).
        price_per_1k_output: USD per 1000 output tokens (default: 0.6e-3).
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm_client: OpenAIClient,
        top_k: int = 5,
        hops: int = 2,
        price_per_1k_input: float = 0.00015,
        price_per_1k_output: float = 0.0006,
        skip_generation: bool = False,
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.top_k = top_k
        self.hops = hops
        self.price_per_1k_input = price_per_1k_input
        self.price_per_1k_output = price_per_1k_output
        self.skip_generation = skip_generation
        self.retriever = GraphRetriever(graph, llm_client)
        self.ranker = SubgraphRanker(graph)

    def evaluate_single(
        self,
        example: EvalExample,
        usage_tracker: _UsageTrackingClient,
    ) -> EvalResult:
        """Evaluate a single example.

        Args:
            example: The evaluation example.
            usage_tracker: Token usage tracker wrapping the LLM client.

        Returns:
            EvalResult with metrics, latency, and cost.
        """
        start_time = time.monotonic()

        # Perform retrieval
        retrieval_result = self.retriever.retrieve(
            query=example.query,
            top_k=self.top_k,
            hops=self.hops,
        )

        # Rank retrieved entities for top-k list
        ranked_entities = self.ranker.rank_and_filter(
            retrieval_result.entities,
            example.query,
            self.top_k,
        )

        if not self.skip_generation:
            user_prompt = build_response_prompt(
                example.query,
                retrieval_result.context_text,
            )
            _ = self.llm_client.chat(
                system_prompt=RESPONSE_GENERATION_SYSTEM,
                user_prompt=user_prompt,
            )

        elapsed = time.monotonic() - start_time

        # Collect retrieved entity names in ranked order
        retrieved_names = [e.name for e in ranked_entities]

        # Compute metrics against reference entities
        reference = example.reference_entities
        precision = compute_precision_at_k(retrieved_names, reference, self.top_k)
        recall = compute_recall_at_k(retrieved_names, reference, self.top_k)
        mrr = compute_mrr(retrieved_names, reference)
        ndcg = compute_ndcg_at_k(retrieved_names, reference, self.top_k)

        # Get token usage
        usage = usage_tracker.get_and_reset_usage()

        # Estimate cost
        cost = (
            (usage.prompt_tokens / 1000.0) * self.price_per_1k_input
            + (usage.completion_tokens / 1000.0) * self.price_per_1k_output
        )

        return EvalResult(
            example_id=example.id or f"example_{id(example)}",
            query=example.query,
            reference_entities=reference,
            retrieved_entities=retrieved_names,
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg,
            latency_seconds=round(elapsed, 4),
            token_usage=usage,
            estimated_cost_usd=round(cost, 6),
            tags=example.tags,
        )

    def run(self, dataset_path: str | Path) -> tuple[list[EvalResult], EvalSummary]:
        """Run evaluation over an entire dataset.

        Args:
            dataset_path: Path to JSONL dataset file.

        Returns:
            Tuple of (per-example results, aggregated summary).
        """
        examples = load_dataset(dataset_path)
        return self.run_examples(examples)

    def run_examples(
        self, examples: list[EvalExample]
    ) -> tuple[list[EvalResult], EvalSummary]:
        """Run evaluation over a list of examples.

        Args:
            examples: List of evaluation examples.

        Returns:
            Tuple of (per-example results, aggregated summary).
        """
        if not examples:
            return [], EvalSummary(
                num_examples=0,
                k=self.top_k,
                avg_precision_at_k=0.0,
                avg_recall_at_k=0.0,
                avg_mrr=0.0,
                avg_ndcg_at_k=0.0,
                avg_latency_seconds=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                total_estimated_cost_usd=0.0,
            )

        tracker = _UsageTrackingClient(self.llm_client)
        results: list[EvalResult] = []

        try:
            for example in examples:
                result = self.evaluate_single(example, tracker)
                results.append(result)
        finally:
            tracker.restore()

        summary = self._compute_summary(results)
        return results, summary

    def _compute_summary(self, results: list[EvalResult]) -> EvalSummary:
        """Aggregate per-example results into a summary.

        Args:
            results: List of per-example results.

        Returns:
            Aggregated EvalSummary.
        """
        n = len(results)
        if n == 0:
            return EvalSummary(
                num_examples=0,
                k=self.top_k,
                avg_precision_at_k=0.0,
                avg_recall_at_k=0.0,
                avg_mrr=0.0,
                avg_ndcg_at_k=0.0,
                avg_latency_seconds=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                total_estimated_cost_usd=0.0,
            )

        total_prompt_tokens = sum(r.token_usage.prompt_tokens for r in results)
        total_completion_tokens = sum(
            r.token_usage.completion_tokens for r in results
        )
        total_tokens = sum(r.token_usage.total_tokens for r in results)
        total_cost = (
            (total_prompt_tokens / 1000.0) * self.price_per_1k_input
            + (total_completion_tokens / 1000.0) * self.price_per_1k_output
        )

        return EvalSummary(
            num_examples=n,
            k=self.top_k,
            avg_precision_at_k=round(sum(r.precision_at_k for r in results) / n, 4),
            avg_recall_at_k=round(sum(r.recall_at_k for r in results) / n, 4),
            avg_mrr=round(sum(r.mrr for r in results) / n, 4),
            avg_ndcg_at_k=round(sum(r.ndcg_at_k for r in results) / n, 4),
            avg_latency_seconds=round(sum(r.latency_seconds for r in results) / n, 4),
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_estimated_cost_usd=round(total_cost, 6),
        )


def save_eval_output(
    results: list[EvalResult],
    summary: EvalSummary,
    output_path: str | Path,
) -> None:
    """Save evaluation results and summary to a JSON file.

    Args:
        results: List of per-example results.
        summary: Aggregated summary.
        output_path: Path for the output JSON file.
    """
    output = {
        "summary": summary.to_dict(),
        "results": [r.to_dict() for r in results],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
