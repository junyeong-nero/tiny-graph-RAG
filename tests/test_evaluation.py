"""Tests for the evaluation framework: metrics, dataset loading, and runner."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tiny_graph_rag.evaluation.dataset import EvalExample, load_dataset
from tiny_graph_rag.evaluation.metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from tiny_graph_rag.evaluation.runner import (
    EvalSummary,
    EvaluationRunner,
    TokenUsage,
    save_eval_output,
)
from tiny_graph_rag.graph.models import Entity, KnowledgeGraph, Relationship


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestPrecisionAtK:

    def test_perfect_precision(self):
        retrieved = ["A", "B", "C"]
        relevant = ["A", "B", "C"]
        assert compute_precision_at_k(retrieved, relevant, 3) == 1.0

    def test_no_relevant_retrieved(self):
        retrieved = ["X", "Y", "Z"]
        relevant = ["A", "B"]
        assert compute_precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_match(self):
        retrieved = ["A", "X", "B"]
        relevant = ["A", "B", "C"]
        assert compute_precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self):
        retrieved = ["A"]
        relevant = ["A", "B"]
        assert compute_precision_at_k(retrieved, relevant, 5) == pytest.approx(1 / 5)

    def test_k_zero(self):
        assert compute_precision_at_k(["A"], ["A"], 0) == 0.0

    def test_empty_relevant(self):
        assert compute_precision_at_k(["A", "B"], [], 3) == 0.0

    def test_empty_retrieved(self):
        assert compute_precision_at_k([], ["A", "B"], 3) == 0.0

    def test_case_insensitive(self):
        retrieved = ["alice", "BOB"]
        relevant = ["Alice", "Bob"]
        assert compute_precision_at_k(retrieved, relevant, 2) == 1.0

    def test_duplicates_counted_once(self):
        retrieved = ["A", "A", "B"]
        relevant = ["A", "B"]
        assert compute_precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


class TestRecallAtK:

    def test_perfect_recall(self):
        retrieved = ["A", "B", "C"]
        relevant = ["A", "B"]
        assert compute_recall_at_k(retrieved, relevant, 3) == 1.0

    def test_no_recall(self):
        retrieved = ["X", "Y"]
        relevant = ["A", "B"]
        assert compute_recall_at_k(retrieved, relevant, 2) == 0.0

    def test_partial_recall(self):
        retrieved = ["A", "X", "Y"]
        relevant = ["A", "B"]
        assert compute_recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)

    def test_k_zero(self):
        assert compute_recall_at_k(["A"], ["A"], 0) == 0.0

    def test_empty_relevant(self):
        assert compute_recall_at_k(["A"], [], 3) == 0.0

    def test_case_insensitive(self):
        retrieved = ["alice"]
        relevant = ["Alice", "Bob"]
        assert compute_recall_at_k(retrieved, relevant, 1) == pytest.approx(0.5)

    def test_duplicates_counted_once(self):
        retrieved = ["A", "A", "B"]
        relevant = ["A", "B"]
        assert compute_recall_at_k(retrieved, relevant, 3) == 1.0


class TestMRR:

    def test_first_position(self):
        retrieved = ["A", "B", "C"]
        relevant = ["A"]
        assert compute_mrr(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = ["X", "A", "B"]
        relevant = ["A"]
        assert compute_mrr(retrieved, relevant) == pytest.approx(0.5)

    def test_third_position(self):
        retrieved = ["X", "Y", "A"]
        relevant = ["A"]
        assert compute_mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        retrieved = ["X", "Y", "Z"]
        relevant = ["A"]
        assert compute_mrr(retrieved, relevant) == 0.0

    def test_empty_relevant(self):
        assert compute_mrr(["A", "B"], []) == 0.0

    def test_empty_retrieved(self):
        assert compute_mrr([], ["A"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        retrieved = ["X", "A", "B"]
        relevant = ["A", "B"]
        assert compute_mrr(retrieved, relevant) == pytest.approx(0.5)

    def test_case_insensitive(self):
        retrieved = ["x", "alice"]
        relevant = ["Alice"]
        assert compute_mrr(retrieved, relevant) == pytest.approx(0.5)


class TestNDCGAtK:

    def test_perfect_ndcg(self):
        retrieved = ["A", "B", "C"]
        relevant = ["A", "B", "C"]
        assert compute_ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_no_relevant(self):
        retrieved = ["X", "Y", "Z"]
        relevant = ["A", "B"]
        assert compute_ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_empty_relevant(self):
        assert compute_ndcg_at_k(["A"], [], 3) == 0.0

    def test_k_zero(self):
        assert compute_ndcg_at_k(["A"], ["A"], 0) == 0.0

    def test_single_relevant_at_first(self):
        retrieved = ["A", "X", "Y"]
        relevant = ["A"]
        assert compute_ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_single_relevant_at_second(self):
        retrieved = ["X", "A", "Y"]
        relevant = ["A"]
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert compute_ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected)

    def test_two_relevant_reversed(self):
        retrieved = ["B", "A"]
        relevant = ["A", "B"]
        assert compute_ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)

    def test_case_insensitive(self):
        retrieved = ["alice", "BOB"]
        relevant = ["Alice", "Bob"]
        assert compute_ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)

    def test_duplicates_counted_once(self):
        retrieved = ["A", "A"]
        relevant = ["A"]
        assert compute_ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------


class TestEvalExample:

    def test_from_dict_minimal(self):
        data = {"query": "test?", "reference_entities": ["A", "B"]}
        ex = EvalExample.from_dict(data)
        assert ex.query == "test?"
        assert ex.reference_entities == ["A", "B"]
        assert ex.id == ""
        assert ex.reference_relationships == []
        assert ex.ground_truth == ""
        assert ex.tags == []

    def test_from_dict_full(self):
        data = {
            "id": "e1",
            "query": "q?",
            "reference_entities": ["X"],
            "reference_relationships": [
                {"source": "X", "target": "Y", "type": "REL"}
            ],
            "ground_truth": "answer text",
            "tags": ["t1"],
        }
        ex = EvalExample.from_dict(data)
        assert ex.id == "e1"
        assert ex.ground_truth == "answer text"
        assert ex.tags == ["t1"]

    def test_from_dict_ground_truth_list_takes_first(self):
        data = {
            "query": "q?",
            "reference_entities": ["A"],
            "ground_truth": ["first answer", "second"],
        }
        ex = EvalExample.from_dict(data)
        assert ex.ground_truth == "first answer"

    def test_from_dict_missing_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            EvalExample.from_dict({"reference_entities": ["A"]})

    def test_from_dict_empty_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            EvalExample.from_dict({"query": " ", "reference_entities": ["A"]})

    def test_from_dict_non_string_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            EvalExample.from_dict({"query": 123, "reference_entities": ["A"]})

    def test_from_dict_missing_entities_raises(self):
        with pytest.raises(ValueError, match="reference_entities"):
            EvalExample.from_dict({"query": "test?"})

    def test_from_dict_invalid_entities_type_raises(self):
        with pytest.raises(ValueError, match="reference_entities"):
            EvalExample.from_dict({"query": "q?", "reference_entities": "A"})

    def test_from_dict_invalid_relationships_type_raises(self):
        with pytest.raises(ValueError, match="reference_relationships"):
            EvalExample.from_dict(
                {
                    "query": "q?",
                    "reference_entities": ["A"],
                    "reference_relationships": ["REL"],
                }
            )

    def test_to_dict_roundtrip(self):
        ex = EvalExample(
            id="e1",
            query="q?",
            reference_entities=["A"],
            reference_relationships=[],
            ground_truth="g",
            tags=["t"],
        )
        data = ex.to_dict()
        restored = EvalExample.from_dict(data)
        assert restored.id == ex.id
        assert restored.query == ex.query
        assert restored.reference_entities == ex.reference_entities

    def test_to_dict_omits_empty_optional(self):
        ex = EvalExample(query="q?", reference_entities=["A"])
        data = ex.to_dict()
        assert "id" not in data
        assert "reference_relationships" not in data
        assert "ground_truth" not in data
        assert "tags" not in data


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestLoadDataset:

    def test_load_valid_jsonl(self, tmp_path: Path):
        jsonl_path = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"query": "q1?", "reference_entities": ["A"]}),
            json.dumps({"id": "e2", "query": "q2?", "reference_entities": ["B", "C"]}),
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")

        examples = load_dataset(jsonl_path)
        assert len(examples) == 2
        assert examples[0].query == "q1?"
        assert examples[1].id == "e2"

    def test_load_skips_blank_lines(self, tmp_path: Path):
        jsonl_path = tmp_path / "test.jsonl"
        content = (
            json.dumps({"query": "q?", "reference_entities": ["A"]})
            + "\n\n\n"
            + json.dumps({"query": "q2?", "reference_entities": ["B"]})
        )
        jsonl_path.write_text(content, encoding="utf-8")

        examples = load_dataset(jsonl_path)
        assert len(examples) == 2

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.jsonl")

    def test_load_invalid_json_raises(self, tmp_path: Path):
        jsonl_path = tmp_path / "bad.jsonl"
        jsonl_path.write_text("not valid json\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_dataset(jsonl_path)

    def test_load_missing_field_raises(self, tmp_path: Path):
        jsonl_path = tmp_path / "bad.jsonl"
        jsonl_path.write_text(json.dumps({"query": "q?"}) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="reference_entities"):
            load_dataset(jsonl_path)

    def test_load_sample_dataset(self):
        sample_path = Path(__file__).parent.parent / "data" / "eval_sample.jsonl"
        if not sample_path.exists():
            pytest.skip("Sample dataset not found")
        examples = load_dataset(sample_path)
        assert len(examples) == 5
        assert all(ex.query for ex in examples)
        assert all(ex.reference_entities for ex in examples)


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:

    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.total_tokens == 0


# ---------------------------------------------------------------------------
# EvalSummary
# ---------------------------------------------------------------------------


class TestEvalSummary:

    def test_to_dict(self):
        s = EvalSummary(
            num_examples=3,
            k=5,
            avg_precision_at_k=0.6,
            avg_recall_at_k=0.5,
            avg_mrr=0.8,
            avg_ndcg_at_k=0.7,
            avg_latency_seconds=0.123,
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_tokens=150,
            total_estimated_cost_usd=0.001,
        )
        d = s.to_dict()
        assert d["num_examples"] == 3
        assert d["k"] == 5
        assert d["total_tokens"] == 150


# ---------------------------------------------------------------------------
# save_eval_output
# ---------------------------------------------------------------------------


class TestSaveEvalOutput:

    def test_save_creates_json(self, tmp_path: Path):
        summary = EvalSummary(
            num_examples=0, k=5, avg_precision_at_k=0.0,
            avg_recall_at_k=0.0, avg_mrr=0.0, avg_ndcg_at_k=0.0,
            avg_latency_seconds=0.0, total_prompt_tokens=0,
            total_completion_tokens=0, total_tokens=0,
            total_estimated_cost_usd=0.0,
        )
        out_path = tmp_path / "output.json"
        save_eval_output([], summary, out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "summary" in data
        assert "results" in data
        assert data["results"] == []


# ---------------------------------------------------------------------------
# EvaluationRunner (mocked LLM)
# ---------------------------------------------------------------------------


class TestEvaluationRunner:

    @pytest.fixture
    def sample_graph(self):
        graph = KnowledgeGraph()
        e_john = Entity(name="John", entity_type="PERSON", description="An engineer")
        e_acme = Entity(
            name="Acme Corp", entity_type="ORGANIZATION", description="A tech company"
        )
        id_john = graph.add_entity(e_john)
        id_acme = graph.add_entity(e_acme)
        graph.add_relationship(Relationship(
            source_entity_id=id_john,
            target_entity_id=id_acme,
            relationship_type="WORKS_FOR",
            description="John works for Acme Corp",
        ))
        return graph

    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.client = MagicMock()

        mock.chat_json.return_value = {"entities": ["John", "Acme Corp"]}
        mock.chat.return_value = "John works at Acme Corp."

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"entities": ["John"]}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 70
        mock.client.chat.completions.create = MagicMock(return_value=mock_response)

        return mock

    def test_run_examples_returns_results_and_summary(self, sample_graph, mock_llm):
        runner = EvaluationRunner(
            graph=sample_graph,
            llm_client=mock_llm,
            top_k=5,
            hops=2,
        )
        examples = [
            EvalExample(
                id="t1",
                query="Who does John work for?",
                reference_entities=["John", "Acme Corp"],
            ),
        ]
        results, summary = runner.run_examples(examples)

        assert len(results) == 1
        assert results[0].example_id == "t1"
        assert summary.num_examples == 1
        assert summary.k == 5
        assert results[0].latency_seconds >= 0.0

    def test_empty_examples_returns_empty_summary(self, sample_graph, mock_llm):
        runner = EvaluationRunner(graph=sample_graph, llm_client=mock_llm)
        results, summary = runner.run_examples([])
        assert results == []
        assert summary.num_examples == 0

    def test_result_serialization(self, sample_graph, mock_llm):
        runner = EvaluationRunner(
            graph=sample_graph,
            llm_client=mock_llm,
            top_k=5,
        )
        examples = [
            EvalExample(
                id="ser1",
                query="Who is John?",
                reference_entities=["John"],
                tags=["person"],
            ),
        ]
        results, summary = runner.run_examples(examples)
        d = results[0].to_dict()
        assert "metrics" in d
        assert "token_usage" in d
        assert d["tags"] == ["person"]
