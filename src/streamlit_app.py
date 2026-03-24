"""Streamlit app for Tiny-Graph-RAG visualization and interaction."""

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from tiny_graph_rag import GraphRAG
from tiny_graph_rag.graph.storage import GraphStorage
from tiny_graph_rag.graph.models import KnowledgeGraph
from tiny_graph_rag.visualization import PyVisVisualizer


# ── Palette ───────────────────────────────────────────────────────────────────
BG_BASE        = "#1A1A1A"
BG_SURFACE     = "#242424"
BG_CARD        = "#2E2E2E"
BORDER_COLOR   = "#383838"
TEXT_PRIMARY   = "#E0E0E0"
TEXT_SECONDARY = "#9E9E9E"
ACCENT         = "#7BB8D4"   # pastel blue — primary point colour
ACCENT_LIGHT   = "#A8D4EC"
ACCENT_BORDER  = "#5AA8CC"

ENTITY_COLORS = {
    "PERSON":       ACCENT,       # pastel blue
    "ORGANIZATION": "#82C4A0",    # soft mint
    "PLACE":        "#E8C07A",    # soft amber
    "CONCEPT":      "#B8A0D4",    # soft lavender
    "EVENT":        "#E89090",    # soft rose
    "OTHER":        "#8C8C8C",    # mid grey
}

ENTITY_LABELS_KO = {
    "PERSON": "인물",
    "ORGANIZATION": "조직",
    "PLACE": "장소",
    "CONCEPT": "개념",
    "EVENT": "사건",
    "OTHER": "기타",
}

SELECTED_COLOR        = ACCENT_LIGHT
SELECTED_BORDER_COLOR = ACCENT_BORDER

EDGE_COLORS = {
    "default":   "#505050",
    "highlight": ACCENT,
}

APP_CSS = f"""
<style>
/* ── Base ── */
.stApp {{
    background-color: {BG_BASE};
    color: {TEXT_PRIMARY};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {BG_SURFACE};
    border-right: 1px solid {BORDER_COLOR};
}}
[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── Header / Title ── */
h1, h2, h3 {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600;
}}
h1 {{ letter-spacing: -0.5px; }}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
    border-bottom: 1px solid {BORDER_COLOR};
    gap: 4px;
}}
[data-testid="stTabs"] [role="tab"] {{
    color: {TEXT_SECONDARY} !important;
    background: transparent;
    border-radius: 6px 6px 0 0;
    padding: 6px 16px;
    font-size: 14px;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}

/* ── Buttons ── */
[data-testid="stButton"] > button,
[data-testid="stDownloadButton"] > button {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    transition: border-color 0.15s, color 0.15s;
}}
[data-testid="stButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
    background-color: {BG_CARD};
}}
[data-testid="stButton"] > button[kind="primary"] {{
    background-color: {ACCENT};
    color: #111;
    border: none;
    font-weight: 600;
}}
[data-testid="stButton"] > button[kind="primary"]:hover {{
    background-color: {ACCENT_LIGHT};
    color: #111;
}}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border-color: {ACCENT};
    box-shadow: 0 0 0 2px {ACCENT}33;
}}

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] {{
    color: {ACCENT} !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background-color: {ACCENT} !important;
}}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
    background-color: {BG_CARD};
    border-color: {BORDER_COLOR};
}}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
    background-color: {ACCENT}33;
    color: {ACCENT_LIGHT};
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 12px 16px;
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT} !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY} !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 8px;
}}
[data-testid="stExpander"] summary {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    background-color: {BG_CARD};
    border-color: {BORDER_COLOR};
    color: {TEXT_PRIMARY};
}}

/* ── Divider ── */
hr {{
    border-color: {BORDER_COLOR} !important;
}}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {{
    color: {TEXT_SECONDARY} !important;
}}
</style>
"""


def load_graph(graph_path: str) -> KnowledgeGraph:
    storage = GraphStorage()
    return storage.load_json(graph_path)


def _compute_degrees(
    graph: KnowledgeGraph, filtered_entities: dict,
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

    filtered_entities = {
        entity_id: entity
        for entity_id, entity in graph.entities.items()
        if not filter_types or entity.entity_type in filter_types
    }

    degrees = _compute_degrees(graph, filtered_entities)

    if len(filtered_entities) > max_nodes:
        sorted_ids = sorted(
            filtered_entities.keys(), key=lambda x: degrees.get(x, 0), reverse=True
        )
        limited_ids = set(sorted_ids[:max_nodes])
        filtered_entities = {
            k: v for k, v in filtered_entities.items() if k in limited_ids
        }

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
                font={"size": 14, "color": TEXT_PRIMARY, "strokeWidth": 3, "strokeColor": BG_BASE},
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
                    font={"size": 8, "color": TEXT_SECONDARY, "strokeWidth": 0, "align": "middle"},
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

    neighbor_ids: set[str] = set()
    relevant_rels = []
    for rel in graph.relationships:
        if rel.source_entity_id == center_entity_id:
            neighbor_ids.add(rel.target_entity_id)
            relevant_rels.append(rel)
        elif rel.target_entity_id == center_entity_id:
            neighbor_ids.add(rel.source_entity_id)
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
                font={"size": 16 if is_center else 14, "color": TEXT_PRIMARY, "strokeWidth": 3, "strokeColor": BG_BASE},
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
                    font={"size": 11, "color": TEXT_SECONDARY, "strokeWidth": 2, "strokeColor": BG_BASE, "align": "middle"},
                )
            )

    return nodes, edges


def _darken(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
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

        st.header("Visualization Options")

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
            f"vertical-align:middle;'></span>"
            f"<span style='font-size:12px;color:{TEXT_SECONDARY};vertical-align:middle;'>"
            f"{label}</span>",
            unsafe_allow_html=True,
        )


def render_stats(stats_placeholder, graph: KnowledgeGraph):
    with stats_placeholder:
        col1, col2 = st.columns(2)
        col1.metric("Entities", len(graph.entities))
        col2.metric("Relationships", len(graph.relationships))

        type_counts = {}
        for entity in graph.entities.values():
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

        st.markdown("**Entity Types:**")
        for etype, count in sorted(type_counts.items()):
            color = ENTITY_COLORS.get(etype, ENTITY_COLORS["OTHER"])
            label = ENTITY_LABELS_KO.get(etype, etype)
            st.markdown(
                f"<span style='color:{color};font-weight:bold;'>"
                f"{label}</span> ({etype}): {count}",
                unsafe_allow_html=True,
            )


def generate_graph_html(graph: KnowledgeGraph, filter_types: list[str], max_nodes: int) -> str:
    viz = PyVisVisualizer(graph=graph, filter_types=filter_types or None, max_nodes=max_nodes)
    viz.generate()
    return viz.network.generate_html()


def render_graph_view(graph: KnowledgeGraph, selected_types: list[str], max_nodes: int):
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
            st.markdown(f"**Subgraph: {center_name}**")

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
        highlightColor="#F7A7A6",
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
    entity_color = ENTITY_COLORS.get(entity.entity_type, ENTITY_COLORS["OTHER"])
    type_label = ENTITY_LABELS_KO.get(entity.entity_type, entity.entity_type)

    st.divider()

    st.markdown(
        f"### <span style='color:{entity_color};'>{entity.name}</span>"
        f" <span style='font-size:14px;color:#999;'>{type_label} ({entity.entity_type})</span>",
        unsafe_allow_html=True,
    )

    if entity.aliases:
        alias_tags = " ".join(
            f"<code style='background:{BG_CARD};padding:2px 6px;border-radius:4px;"
            f"font-size:12px;color:{TEXT_SECONDARY};border:1px solid {BORDER_COLOR};'>{alias}</code>"
            for alias in entity.aliases
        )
        st.markdown(f"Aliases: {alias_tags}", unsafe_allow_html=True)

    if entity.description:
        st.markdown(f"> {entity.description}")

    col1, col2 = st.columns(2)

    with col1:
        if details["outgoing"]:
            st.markdown("**Outgoing**")
            for rel in details["outgoing"][:8]:
                target_color = ENTITY_COLORS.get(rel.get("target_type", "OTHER"), ENTITY_COLORS["OTHER"])
                st.markdown(
                    f"- `{rel['type']}` → "
                    f"<span style='color:{target_color};'>{rel['target']}</span>",
                    unsafe_allow_html=True,
                )

    with col2:
        if details["incoming"]:
            st.markdown("**Incoming**")
            for rel in details["incoming"][:8]:
                source_color = ENTITY_COLORS.get(rel.get("source_type", "OTHER"), ENTITY_COLORS["OTHER"])
                st.markdown(
                    f"- <span style='color:{source_color};'>{rel['source']}</span>"
                    f" → `{rel['type']}`",
                    unsafe_allow_html=True,
                )


def render_query_view():
    st.subheader("Query the Knowledge Graph")

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
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")


def render_entity_list(graph: KnowledgeGraph, selected_types: list[str]):
    st.subheader("Entity List")

    search = st.text_input("Search entities", placeholder="Type to filter...")

    filtered = [
        entity
        for entity in graph.entities.values()
        if (not selected_types or entity.entity_type in selected_types)
        and (not search or search.lower() in entity.name.lower())
    ]
    filtered.sort(key=lambda x: x.name)

    st.caption(f"Found {len(filtered)} entities")

    display_limit = 50
    for entity in filtered[:display_limit]:
        type_label = ENTITY_LABELS_KO.get(entity.entity_type, entity.entity_type)

        with st.expander(f"{entity.name}  [{type_label}]"):
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
                            target_color = ENTITY_COLORS.get(target.entity_type, ENTITY_COLORS["OTHER"])
                            st.markdown(
                                f"- `{rel.relationship_type}` → "
                                f"<span style='color:{target_color};'>{target.name}</span>",
                                unsafe_allow_html=True,
                            )
                    else:
                        source = graph.get_entity(rel.source_entity_id)
                        if source:
                            source_color = ENTITY_COLORS.get(source.entity_type, ENTITY_COLORS["OTHER"])
                            st.markdown(
                                f"- <span style='color:{source_color};'>{source.name}</span>"
                                f" → `{rel.relationship_type}`",
                                unsafe_allow_html=True,
                            )

    if len(filtered) > display_limit:
        st.info(
            f"Showing first {display_limit} of {len(filtered)} entities. Use search to filter."
        )


def render_welcome_screen():
    st.info(
        """
    ### Getting Started

    1. **Load a graph**: Enter the path to your knowledge graph JSON file in the sidebar and click "Load Graph"

    2. **Explore**: Use the Graph View tab to visualize and interact with the knowledge graph

    3. **Query**: Use the Query tab to ask questions about the knowledge graph

    4. **Filter**: Use the sidebar options to filter by entity type and limit the number of displayed nodes

    ---

    **Tip**: If you don't have a graph yet, process a document first using the CLI:
    ```bash
    python main.py process your_document.txt -o graph.json
    ```
    """
    )


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
        page_icon="🔗",
        layout="wide",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        f"<h1 style='margin-bottom:2px;'>Tiny-Graph-RAG Visualizer</h1>"
        f"<p style='color:{TEXT_SECONDARY};margin-top:0;font-size:14px;'>"
        f"Interactive knowledge graph visualization and querying</p>",
        unsafe_allow_html=True,
    )

    graph_path, load_button, selected_types, max_nodes, stats_placeholder = (
        render_sidebar()
    )

    init_session_state()

    if load_button:
        try:
            with st.spinner("Loading graph..."):
                st.session_state.graph = load_graph(graph_path)
                st.session_state.rag = GraphRAG()
                st.session_state.rag.load_graph(graph_path)
            st.success("Graph loaded successfully!")
        except FileNotFoundError:
            st.error(f"File not found: {graph_path}")
        except Exception as e:
            st.error(f"Error loading graph: {e}")

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

    st.divider()
    st.markdown(
        f"<div style='text-align:center;color:{TEXT_SECONDARY};font-size:12px;'>"
        f"Tiny-Graph-RAG Visualizer &nbsp;·&nbsp; Built with Streamlit"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
