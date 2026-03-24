"""Streamlit app for Tiny-Graph-RAG visualization and interaction."""

from __future__ import annotations

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from tiny_graph_rag import GraphRAG
from tiny_graph_rag.graph.models import KnowledgeGraph
from tiny_graph_rag.graph.storage import GraphStorage


# -- Palette -----------------------------------------------------------------
BG_BASE = "#0E131B"
BG_SURFACE = "#F4F7FC"
BG_CARD = "#171D27"
BORDER_COLOR = "#2A3342"
TEXT_PRIMARY = "#EDF2FC"
TEXT_SECONDARY = "#A7B3C8"
ACCENT = "#3B6FC7"
ACCENT_LIGHT = "#E8F0FF"
ACCENT_BORDER = "#9CB5E6"

ENTITY_COLORS = {
    "PERSON": "#7DB1FF",
    "ORGANIZATION": "#7AD2B6",
    "PLACE": "#E7C36E",
    "CONCEPT": "#B9A5FF",
    "EVENT": "#F28B82",
    "OTHER": "#9BA8BF",
}

ENTITY_LABELS_KO = {
    "PERSON": "인물",
    "ORGANIZATION": "조직",
    "PLACE": "장소",
    "CONCEPT": "개념",
    "EVENT": "사건",
    "OTHER": "기타",
}

SELECTED_COLOR = ACCENT_LIGHT
SELECTED_BORDER_COLOR = ACCENT_BORDER

EDGE_COLORS = {
    "default": "#526079",
    "highlight": ACCENT,
}

APP_CSS = ""


def load_graph(graph_path: str) -> KnowledgeGraph:
    """Load a graph from JSON storage."""
    storage = GraphStorage()
    return storage.load_json(graph_path)


def _compute_degrees(
    graph: KnowledgeGraph,
    filtered_entities: dict[str, object],
) -> dict[str, int]:
    degrees = {entity_id: 0 for entity_id in filtered_entities}
    for rel in graph.relationships:
        if (
            rel.source_entity_id in filtered_entities
            and rel.target_entity_id in filtered_entities
        ):
            degrees[rel.source_entity_id] = degrees.get(rel.source_entity_id, 0) + 1
            degrees[rel.target_entity_id] = degrees.get(rel.target_entity_id, 0) + 1
    return degrees


def summarize_graph(graph: KnowledgeGraph) -> dict:
    """Return the core graph metrics used in the app and tests."""
    type_counts: dict[str, int] = {}
    for entity in graph.entities.values():
        type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

    entity_count = len(graph.entities)
    relationship_count = len(graph.relationships)
    avg_degree = round((relationship_count * 2) / entity_count, 2) if entity_count else 0.0
    return {
        "entities": entity_count,
        "relationships": relationship_count,
        "avg_degree": avg_degree,
        "type_counts": type_counts,
    }


def select_entity_ids(
    graph: KnowledgeGraph,
    selected_types: list[str] | None,
    max_nodes: int = 200,
) -> set[str]:
    """Return the highest-degree entity ids after type filtering."""
    filtered_entities = {
        entity_id: entity
        for entity_id, entity in graph.entities.items()
        if not selected_types or entity.entity_type in selected_types
    }
    degrees = _compute_degrees(graph, filtered_entities)
    ranked_ids = sorted(
        filtered_entities.keys(),
        key=lambda entity_id: (
            -degrees.get(entity_id, 0),
            graph.entities[entity_id].name.lower(),
            entity_id,
        ),
    )
    return set(ranked_ids[:max_nodes])


def build_focus_entity_ids(graph: KnowledgeGraph, entity_id: str | None) -> set[str]:
    """Return the selected entity and its direct neighbors."""
    if not entity_id or entity_id not in graph.entities:
        return set()

    focused_ids = {entity_id}
    for rel in graph.relationships:
        if rel.source_entity_id == entity_id:
            focused_ids.add(rel.target_entity_id)
        elif rel.target_entity_id == entity_id:
            focused_ids.add(rel.source_entity_id)
    return focused_ids


def _build_node_title(entity, degree: int) -> str:
    lines = [
        f"{entity.name}",
        f"Type: {entity.entity_type}",
        f"Connections: {degree}",
    ]
    if entity.aliases:
        lines.append(f"Aliases: {', '.join(entity.aliases)}")
    if entity.description:
        desc = entity.description[:150]
        if len(entity.description) > 150:
            desc += "..."
        lines.append(f"\n{desc}")
    return "\n".join(lines)


def create_agraph_data(
    graph: KnowledgeGraph,
    filter_types: list[str] | None = None,
    max_nodes: int = 200,
    selected_entity_id: str | None = None,
) -> tuple[list[Node], list[Edge]]:
    nodes = []
    edges = []

    limited_ids = select_entity_ids(graph, filter_types, max_nodes=max_nodes)
    filtered_entities = {
        entity_id: entity
        for entity_id, entity in graph.entities.items()
        if entity_id in limited_ids
    }
    degrees = _compute_degrees(graph, filtered_entities)

    for entity_id, entity in filtered_entities.items():
        base_color = ENTITY_COLORS.get(entity.entity_type, ENTITY_COLORS["OTHER"])
        degree = degrees.get(entity_id, 0)
        size = min(15 + degree * 3, 40)
        is_selected = selected_entity_id and entity_id == selected_entity_id

        node_color = {
            "background": SELECTED_COLOR if is_selected else base_color,
            "border": SELECTED_BORDER_COLOR if is_selected else _darken(base_color),
            "highlight": {
                "background": SELECTED_COLOR,
                "border": SELECTED_BORDER_COLOR,
            },
        }

        if is_selected:
            size = int(size * 1.3)

        title = _build_node_title(entity, degree)
        nodes.append(
            Node(
                id=entity_id,
                label=entity.name,
                size=size,
                color=node_color,
                title=title,
                shape="dot",
                borderWidth=2,
                borderWidthSelected=3,
                font={
                    "size": 14,
                    "color": TEXT_PRIMARY,
                    "strokeWidth": 3,
                    "strokeColor": BG_BASE,
                },
            )
        )

    entity_ids = set(filtered_entities.keys())
    for rel in graph.relationships:
        if rel.source_entity_id in entity_ids and rel.target_entity_id in entity_ids:
            edge_title = rel.description if rel.description else rel.relationship_type
            edges.append(
                Edge(
                    source=rel.source_entity_id,
                    target=rel.target_entity_id,
                    label=rel.relationship_type,
                    color=EDGE_COLORS["default"],
                    title=edge_title,
                    width=1.5,
                    font={
                        "size": 8,
                        "color": TEXT_SECONDARY,
                        "strokeWidth": 0,
                        "align": "middle",
                    },
                )
            )

    return nodes, edges


def create_subgraph_data(
    graph: KnowledgeGraph,
    center_entity_id: str,
) -> tuple[list[Node], list[Edge]]:
    center_entity = graph.get_entity(center_entity_id)
    if not center_entity:
        return [], []

    neighbor_ids = build_focus_entity_ids(graph, center_entity_id) - {center_entity_id}
    relevant_rels = []
    for rel in graph.relationships:
        if rel.source_entity_id == center_entity_id or rel.target_entity_id == center_entity_id:
            relevant_rels.append(rel)

    subgraph_ids = {center_entity_id} | neighbor_ids
    subgraph_entities = {}
    for eid in subgraph_ids:
        entity = graph.get_entity(eid)
        if entity is not None:
            subgraph_entities[eid] = entity

    degrees = _compute_degrees(graph, subgraph_entities)

    nodes = []
    for entity_id, entity in subgraph_entities.items():
        is_center = entity_id == center_entity_id
        base_color = ENTITY_COLORS.get(entity.entity_type, ENTITY_COLORS["OTHER"])
        degree = degrees.get(entity_id, 0)
        size = min(15 + degree * 3, 40)

        node_color = {
            "background": SELECTED_COLOR if is_center else base_color,
            "border": SELECTED_BORDER_COLOR if is_center else _darken(base_color),
            "highlight": {
                "background": SELECTED_COLOR,
                "border": SELECTED_BORDER_COLOR,
            },
        }

        if is_center:
            size = int(size * 1.5)

        title = _build_node_title(entity, degree)
        nodes.append(
            Node(
                id=entity_id,
                label=entity.name,
                size=size,
                color=node_color,
                title=title,
                shape="dot",
                borderWidth=3 if is_center else 2,
                borderWidthSelected=3,
                font={
                    "size": 16 if is_center else 14,
                    "color": TEXT_PRIMARY,
                    "strokeWidth": 3,
                    "strokeColor": BG_BASE,
                },
            )
        )

    edges = []
    for rel in relevant_rels:
        if rel.source_entity_id in subgraph_entities and rel.target_entity_id in subgraph_entities:
            edge_title = rel.description if rel.description else rel.relationship_type
            edges.append(
                Edge(
                    source=rel.source_entity_id,
                    target=rel.target_entity_id,
                    label=rel.relationship_type,
                    color=EDGE_COLORS["default"],
                    title=edge_title,
                    width=2,
                    font={
                        "size": 11,
                        "color": TEXT_SECONDARY,
                        "strokeWidth": 2,
                        "strokeColor": BG_BASE,
                        "align": "middle",
                    },
                )
            )

    return nodes, edges


def _darken(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    factor = 0.7
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


def get_entity_details(graph: KnowledgeGraph, entity_id: str) -> dict:
    entity = graph.get_entity(entity_id)
    if not entity:
        return {}

    relationships = graph.get_relationships_for_entity(entity_id)
    outgoing = []
    incoming = []
    for rel in relationships:
        if rel.source_entity_id == entity_id:
            target = graph.get_entity(rel.target_entity_id)
            if target:
                outgoing.append(
                    {
                        "type": rel.relationship_type,
                        "target": target.name,
                        "target_type": target.entity_type,
                        "description": rel.description,
                    }
                )
        else:
            source = graph.get_entity(rel.source_entity_id)
            if source:
                incoming.append(
                    {
                        "type": rel.relationship_type,
                        "source": source.name,
                        "source_type": source.entity_type,
                        "description": rel.description,
                    }
                )

    return {
        "entity": entity,
        "outgoing": outgoing,
        "incoming": incoming,
    }


def get_entity_panel_data(graph: KnowledgeGraph, entity_id: str | None) -> dict:
    """Return entity panel data with stable keys used by tests."""
    if not entity_id:
        return {"entity": None, "outgoing": [], "incoming": []}

    details = get_entity_details(graph, entity_id)
    if not details:
        return {"entity": None, "outgoing": [], "incoming": []}

    outgoing = [
        {
            "target": item["target"],
            "relationship_type": item["type"],
            "description": item["description"],
        }
        for item in details["outgoing"]
    ]
    incoming = [
        {
            "source": item["source"],
            "relationship_type": item["type"],
            "description": item["description"],
        }
        for item in details["incoming"]
    ]
    return {
        "entity": details["entity"],
        "outgoing": outgoing,
        "incoming": incoming,
    }


def render_section_heading(index: str, title: str):
    del index
    st.subheader(title)


def render_app_heading():
    st.title("Tiny-Graph-RAG Visualizer")
    st.caption("Interactive knowledge graph visualization and querying")


def render_sidebar():
    with st.sidebar:
        st.header("Load Graph")

        graph_path = st.text_input(
            "Graph JSON Path",
            value="data/kg/현진건-운수좋은날-KG.json",
            help="Path to the knowledge graph JSON file",
        )

        load_button = st.button("Load Graph", type="primary")

        st.divider()
        st.header("Visualization")

        all_types = list(ENTITY_COLORS.keys())
        selected_types = st.multiselect(
            "Filter Entity Types",
            options=all_types,
            default=all_types,
            help="Select which entity types to display",
        )

        max_nodes = st.slider(
            "Max Nodes",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Maximum number of nodes to display",
        )

        st.divider()
        st.header("Graph Stats")
        stats_placeholder = st.empty()

    return graph_path, load_button, selected_types, max_nodes, stats_placeholder


def render_legend():
    cols = st.columns(len(ENTITY_COLORS))
    for col, (etype, color) in zip(cols, ENTITY_COLORS.items()):
        label = ENTITY_LABELS_KO.get(etype, etype)
        col.markdown(
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{color};margin-right:6px;"
            f"vertical-align:middle;'></span>{label}",
            unsafe_allow_html=True,
        )


def render_stats(stats_placeholder, graph: KnowledgeGraph):
    with stats_placeholder:
        summary = summarize_graph(graph)
        col1, col2 = st.columns(2)
        col1.metric("Entities", summary["entities"])
        col2.metric("Relationships", summary["relationships"])
        st.metric("Avg Degree", summary["avg_degree"])

        st.markdown("**Entity Types:**")
        for etype, count in sorted(summary["type_counts"].items()):
            color = ENTITY_COLORS.get(etype, ENTITY_COLORS["OTHER"])
            label = ENTITY_LABELS_KO.get(etype, etype)
            st.markdown(
                f"<span style='color:{color};font-weight:bold;'>{label}</span> ({etype}): {count}",
                unsafe_allow_html=True,
            )


def generate_graph_html(graph: KnowledgeGraph, filter_types: list[str], max_nodes: int) -> str:
    from tiny_graph_rag.visualization import PyVisVisualizer

    viz = PyVisVisualizer(graph=graph, filter_types=filter_types or None, max_nodes=max_nodes)
    viz.generate()
    return viz.network.generate_html()


def render_graph_view(graph: KnowledgeGraph, selected_types: list[str], max_nodes: int):
    render_section_heading("01", "Graph View")
    render_legend()

    is_subgraph = st.session_state.subgraph_center is not None
    if is_subgraph:
        center_id = st.session_state.subgraph_center
        center_entity = graph.get_entity(center_id)
        center_name = center_entity.name if center_entity else center_id

        col_back, col_label = st.columns([1, 5])
        with col_back:
            if st.button("< Full Graph"):
                st.session_state.subgraph_center = None
                st.session_state.selected_entity = None
                st.rerun()
        with col_label:
            st.markdown(f"**Focused subgraph:** {center_name}")

        nodes, edges = create_subgraph_data(graph, center_id)
    else:
        nodes, edges = create_agraph_data(
            graph,
            filter_types=selected_types if selected_types else None,
            max_nodes=max_nodes,
            selected_entity_id=st.session_state.selected_entity,
        )

    if not nodes:
        st.warning("No entities match the current filters.")
        return

    col_caption, col_download = st.columns([5, 1])
    with col_caption:
        st.caption(f"{len(nodes)} entities / {len(edges)} relationships")
    with col_download:
        html_content = generate_graph_html(graph, selected_types, max_nodes)
        st.download_button(
            label="Download HTML",
            data=html_content,
            file_name="graph.html",
            mime="text/html",
        )

    config = Config(
        width=1100,
        height=700,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor=ACCENT_LIGHT,
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": True},
    )

    selected = agraph(nodes=nodes, edges=edges, config=config)
    if selected:
        if is_subgraph and selected != st.session_state.subgraph_center:
            st.session_state.subgraph_center = selected
            st.session_state.selected_entity = selected
            st.rerun()
        elif not is_subgraph:
            st.session_state.subgraph_center = selected
            st.session_state.selected_entity = selected
            st.rerun()

    if is_subgraph:
        details = get_entity_details(graph, st.session_state.subgraph_center)
        if details:
            render_entity_detail_card(details)


def render_entity_detail_card(details: dict):
    entity = details["entity"]
    type_label = ENTITY_LABELS_KO.get(entity.entity_type, entity.entity_type)

    st.divider()
    st.subheader(entity.name)
    st.caption(f"{type_label} ({entity.entity_type})")

    if entity.aliases:
        st.markdown(f"**Aliases:** {', '.join(entity.aliases)}")

    if entity.description:
        st.write(entity.description)

    col1, col2 = st.columns(2)

    with col1:
        if details["outgoing"]:
            st.markdown("**Outgoing**")
            for rel in details["outgoing"][:8]:
                st.markdown(f"- `{rel['type']}` → {rel['target']}")

    with col2:
        if details["incoming"]:
            st.markdown("**Incoming**")
            for rel in details["incoming"][:8]:
                st.markdown(f"- {rel['source']} → `{rel['type']}`")


def render_query_view():
    render_section_heading("02", "Query")

    if not st.session_state.rag:
        st.warning("Load a graph first to enable querying.")
        return

    query = st.text_input(
        "Enter your question",
        placeholder="What is the relationship between X and Y?",
    )

    if st.button("Ask", type="primary"):
        if not query:
            st.warning("Please enter a question.")
            return

        with st.spinner("Generating response..."):
            try:
                response = st.session_state.rag.query(query)
                st.markdown("### Response")
                st.write(response)
            except Exception as exc:
                st.error(f"Error: {exc}")


def render_entity_list(graph: KnowledgeGraph, selected_types: list[str]):
    render_section_heading("03", "Entity Archive")

    search = st.text_input("Search entities", placeholder="Type to filter...")
    filtered = [
        entity
        for entity in graph.entities.values()
        if (not selected_types or entity.entity_type in selected_types)
        and (not search or search.lower() in entity.name.lower())
    ]
    filtered.sort(key=lambda item: item.name)

    st.caption(f"Found {len(filtered)} entities")

    display_limit = 50
    for entity in filtered[:display_limit]:
        type_label = ENTITY_LABELS_KO.get(entity.entity_type, entity.entity_type)
        with st.expander(f"{entity.name} [{type_label}]"):
            if entity.aliases:
                st.markdown(f"**Aliases:** {', '.join(entity.aliases)}")
            st.markdown(f"**Description:** {entity.description or 'N/A'}")

            rels = graph.get_relationships_for_entity(entity.entity_id)
            if rels:
                st.markdown("**Relationships:**")
                for rel in rels[:10]:
                    if rel.source_entity_id == entity.entity_id:
                        target = graph.get_entity(rel.target_entity_id)
                        if target:
                            st.markdown(f"- `{rel.relationship_type}` -> {target.name}")
                    else:
                        source = graph.get_entity(rel.source_entity_id)
                        if source:
                            st.markdown(f"- {source.name} -> `{rel.relationship_type}`")

    if len(filtered) > display_limit:
        st.info(f"Showing first {display_limit} of {len(filtered)} entities. Use search to filter.")


def render_welcome_screen():
    render_section_heading("00", "Getting Started")
    st.info("Load a graph from the sidebar to begin.")
    st.code("python main.py process your_document.txt -o graph.json", language="bash")


def init_session_state():
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "selected_entity" not in st.session_state:
        st.session_state.selected_entity = None
    if "subgraph_center" not in st.session_state:
        st.session_state.subgraph_center = None


def main():
    st.set_page_config(
        page_title="Tiny-Graph-RAG Visualizer",
        page_icon=":material/hub:",
        layout="wide",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    render_app_heading()

    graph_path, load_button, selected_types, max_nodes, stats_placeholder = render_sidebar()
    init_session_state()
    load_feedback: tuple[str, str] | None = None

    if load_button:
        try:
            with st.spinner("Loading graph..."):
                st.session_state.graph = load_graph(graph_path)
                st.session_state.rag = GraphRAG()
                st.session_state.rag.load_graph(graph_path)
            load_feedback = ("success", "Graph loaded successfully!")
        except FileNotFoundError:
            load_feedback = ("error", f"File not found: {graph_path}")
        except Exception as exc:
            load_feedback = ("error", f"Error loading graph: {exc}")

    feedback_placeholder = st.empty()
    if load_feedback is not None:
        feedback_type, feedback_message = load_feedback
        if feedback_type == "success":
            feedback_placeholder.success(feedback_message)
        else:
            feedback_placeholder.error(feedback_message)

    if st.session_state.graph:
        graph = st.session_state.graph
        render_stats(stats_placeholder, graph)

        tab1, tab2, tab3 = st.tabs(["Graph View", "Query", "Entity List"])
        with tab1:
            render_graph_view(graph, selected_types, max_nodes)
        with tab2:
            render_query_view()
        with tab3:
            render_entity_list(graph, selected_types)
    else:
        render_welcome_screen()


if __name__ == "__main__":
    main()
