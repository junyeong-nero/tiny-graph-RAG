from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

agraph_stub = types.ModuleType("streamlit_agraph")


class DummyConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DummyNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DummyEdge:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


agraph_stub.Config = DummyConfig
agraph_stub.Node = DummyNode
agraph_stub.Edge = DummyEdge
agraph_stub.agraph = lambda *args, **kwargs: None
sys.modules.setdefault("streamlit_agraph", agraph_stub)

from tiny_graph_rag.graph.models import Entity, KnowledgeGraph, Relationship
from streamlit_app import (
    build_focus_entity_ids,
    get_entity_panel_data,
    select_entity_ids,
    summarize_graph,
)


def build_sample_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()

    kim = Entity(
        entity_id="e1",
        name="Kim",
        entity_type="PERSON",
        description="Main character",
        aliases=["김씨"],
    )
    market = Entity(
        entity_id="e2",
        name="Market",
        entity_type="PLACE",
        description="A crowded market",
    )
    rain = Entity(
        entity_id="e3",
        name="Rain",
        entity_type="EVENT",
        description="Heavy rain starts",
    )
    coat = Entity(
        entity_id="e4",
        name="Coat",
        entity_type="CONCEPT",
        description="An old coat",
    )

    for entity in [kim, market, rain, coat]:
        graph.add_entity(entity)

    graph.add_relationship(
        Relationship(
            source_entity_id="e1",
            target_entity_id="e2",
            relationship_type="VISITS",
        )
    )
    graph.add_relationship(
        Relationship(
            source_entity_id="e1",
            target_entity_id="e3",
            relationship_type="EXPERIENCES",
        )
    )
    graph.add_relationship(
        Relationship(
            source_entity_id="e4",
            target_entity_id="e1",
            relationship_type="BELONGS_TO",
        )
    )
    return graph


def test_summarize_graph_reports_core_counts():
    summary = summarize_graph(build_sample_graph())

    assert summary["entities"] == 4
    assert summary["relationships"] == 3
    assert summary["avg_degree"] == 1.5
    assert summary["type_counts"]["PERSON"] == 1


def test_select_entity_ids_filters_and_caps_by_degree():
    graph = build_sample_graph()

    selected = select_entity_ids(graph, ["PERSON", "PLACE", "EVENT"], max_nodes=2)

    assert selected == {"e1", "e2"}


def test_build_focus_entity_ids_returns_center_and_neighbors():
    graph = build_sample_graph()

    focused = build_focus_entity_ids(graph, "e1")

    assert focused == {"e1", "e2", "e3", "e4"}
    assert build_focus_entity_ids(graph, "missing") == set()


def test_get_entity_panel_data_splits_incoming_and_outgoing():
    graph = build_sample_graph()

    panel = get_entity_panel_data(graph, "e1")

    assert panel["entity"].name == "Kim"
    assert [item["target"] for item in panel["outgoing"]] == ["Market", "Rain"]
    assert [item["source"] for item in panel["incoming"]] == ["Coat"]
