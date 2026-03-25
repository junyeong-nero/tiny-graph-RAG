"""Pure HTML + JavaScript knowledge graph visualizer using vis.js."""

import json
import webbrowser
from collections import defaultdict
from pathlib import Path

from ..graph.models import KnowledgeGraph

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
      background: #1a1a2e;
      color: #e0e0e0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 12px 20px;
      background: #16213e;
      border-bottom: 1px solid #0f3460;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }

    header h1 {
      font-size: 16px;
      font-weight: 600;
      color: #a8c7fa;
      flex-shrink: 0;
    }

    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      flex: 1;
    }

    .controls label {
      font-size: 12px;
      color: #9aa0b4;
    }

    select, input[type=range] {
      background: #0f3460;
      border: 1px solid #1e4d8c;
      border-radius: 4px;
      color: #e0e0e0;
      padding: 3px 8px;
      font-size: 12px;
    }

    input[type=range] { padding: 0; width: 90px; }

    .stats {
      font-size: 11px;
      color: #6b7a99;
      margin-left: auto;
      white-space: nowrap;
    }

    .main {
      flex: 1;
      display: flex;
      overflow: hidden;
    }

    #graph-container {
      flex: 1;
      position: relative;
    }

    #network {
      width: 100%;
      height: 100%;
    }

    #tooltip {
      position: absolute;
      background: rgba(22, 33, 62, 0.95);
      border: 1px solid #0f3460;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 12px;
      max-width: 280px;
      pointer-events: none;
      display: none;
      z-index: 10;
      line-height: 1.6;
    }

    #tooltip .tt-name { font-weight: 700; font-size: 14px; color: #a8c7fa; }
    #tooltip .tt-type { color: #9aa0b4; font-size: 11px; margin-bottom: 6px; }
    #tooltip .tt-desc { color: #c8d0e7; }

    #legend {
      width: 160px;
      background: #16213e;
      border-left: 1px solid #0f3460;
      padding: 14px 12px;
      overflow-y: auto;
      flex-shrink: 0;
    }

    #legend h2 { font-size: 12px; color: #9aa0b4; margin-bottom: 10px; }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 7px;
      font-size: 12px;
      cursor: pointer;
      opacity: 1;
      transition: opacity 0.2s;
    }

    .legend-item.hidden { opacity: 0.3; }

    .legend-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }
  </style>
</head>
<body>

<header>
  <h1>Knowledge Graph</h1>
  <div class="controls">
    <label>Filter type:
      <select id="type-filter">
        <option value="ALL">All</option>
      </select>
    </label>
    <label>Min weight:
      <input type="range" id="weight-filter" min="0" max="10" step="0.5" value="0" />
      <span id="weight-val">0</span>
    </label>
    <label>Max nodes:
      <input type="range" id="node-limit" min="10" max="500" step="10" value="200" />
      <span id="node-val">200</span>
    </label>
  </div>
  <div class="stats" id="stats"></div>
</header>

<div class="main">
  <div id="graph-container">
    <div id="network"></div>
    <div id="tooltip"></div>
  </div>
  <div id="legend">
    <h2>Entity Types</h2>
    <div id="legend-items"></div>
  </div>
</div>

<script>
const RAW_DATA = __GRAPH_DATA__;

const COLORS = {
  PERSON:       { bg: "#3b82f6", border: "#1d4ed8" },
  ORGANIZATION: { bg: "#22c55e", border: "#15803d" },
  PLACE:        { bg: "#f97316", border: "#c2410c" },
  CONCEPT:      { bg: "#a855f7", border: "#7e22ce" },
  EVENT:        { bg: "#ef4444", border: "#b91c1c" },
  OTHER:        { bg: "#6b7280", border: "#374151" },
};

function getColor(type) {
  return COLORS[type] || COLORS.OTHER;
}

// Collect all entity types from data
const allTypes = [...new Set(Object.values(RAW_DATA.entities).map(e => e.entity_type))].sort();

// Populate type filter select
const typeSelect = document.getElementById("type-filter");
allTypes.forEach(t => {
  const opt = document.createElement("option");
  opt.value = t;
  opt.textContent = t;
  typeSelect.appendChild(opt);
});

// Populate legend
const legendContainer = document.getElementById("legend-items");
const hiddenTypes = new Set();
allTypes.forEach(type => {
  const color = getColor(type);
  const item = document.createElement("div");
  item.className = "legend-item";
  item.dataset.type = type;
  item.innerHTML = `<div class="legend-dot" style="background:${color.bg}"></div><span>${type}</span>`;
  item.addEventListener("click", () => {
    if (hiddenTypes.has(type)) {
      hiddenTypes.delete(type);
      item.classList.remove("hidden");
    } else {
      hiddenTypes.add(type);
      item.classList.add("hidden");
    }
    renderGraph();
  });
  legendContainer.appendChild(item);
});

// Range inputs
const weightSlider = document.getElementById("weight-filter");
const weightVal = document.getElementById("weight-val");
const nodeSlider = document.getElementById("node-limit");
const nodeVal = document.getElementById("node-val");

weightSlider.addEventListener("input", () => { weightVal.textContent = weightSlider.value; renderGraph(); });
nodeSlider.addEventListener("input", () => { nodeVal.textContent = nodeSlider.value; renderGraph(); });
typeSelect.addEventListener("change", renderGraph);

let network = null;

function buildGraphData() {
  const filterType = typeSelect.value;
  const minWeight = parseFloat(weightSlider.value);
  const maxNodes = parseInt(nodeSlider.value);

  // Filter entities
  let entityIds = Object.keys(RAW_DATA.entities).filter(id => {
    const e = RAW_DATA.entities[id];
    if (hiddenTypes.has(e.entity_type)) return false;
    if (filterType !== "ALL" && e.entity_type !== filterType) return false;
    return true;
  });

  // Calculate degrees
  const degrees = {};
  entityIds.forEach(id => { degrees[id] = 0; });
  const entitySet = new Set(entityIds);

  RAW_DATA.relationships.forEach(rel => {
    if (rel.weight < minWeight) return;
    if (entitySet.has(rel.source_entity_id)) degrees[rel.source_entity_id] = (degrees[rel.source_entity_id] || 0) + 1;
    if (entitySet.has(rel.target_entity_id)) degrees[rel.target_entity_id] = (degrees[rel.target_entity_id] || 0) + 1;
  });

  // Limit to top N by degree
  if (entityIds.length > maxNodes) {
    entityIds = entityIds.sort((a, b) => (degrees[b] || 0) - (degrees[a] || 0)).slice(0, maxNodes);
  }

  const visibleSet = new Set(entityIds);

  const nodes = entityIds.map(id => {
    const e = RAW_DATA.entities[id];
    const deg = degrees[id] || 0;
    const color = getColor(e.entity_type);
    const size = Math.min(10 + deg * 2, 50);
    return {
      id,
      label: e.name,
      title: buildNodeTooltip(e),
      color: { background: color.bg, border: color.border, highlight: { background: color.bg, border: "#fff" } },
      size,
      font: { color: "#e0e0e0", size: 13 },
      borderWidth: 2,
      _entity: e,
    };
  });

  const edges = RAW_DATA.relationships
    .filter(rel => rel.weight >= minWeight && visibleSet.has(rel.source_entity_id) && visibleSet.has(rel.target_entity_id))
    .map(rel => ({
      from: rel.source_entity_id,
      to: rel.target_entity_id,
      label: rel.relationship_type,
      title: rel.description ? `${rel.relationship_type}: ${rel.description}` : rel.relationship_type,
      width: Math.min(1 + rel.weight * 1.5, 6),
      color: { color: "#4b5563", highlight: "#a8c7fa" },
      arrows: { to: { enabled: true, scaleFactor: 0.6 } },
      font: { color: "#9aa0b4", size: 10, align: "middle" },
      smooth: { type: "continuous" },
    }));

  return { nodes, edges };
}

function buildNodeTooltip(entity) {
  const desc = entity.description
    ? entity.description.slice(0, 200) + (entity.description.length > 200 ? "..." : "")
    : "";
  return `<div class="tt-name">${entity.name}</div>
<div class="tt-type">${entity.entity_type}</div>
${desc ? `<div class="tt-desc">${desc}</div>` : ""}`;
}

function renderGraph() {
  const { nodes, edges } = buildGraphData();

  document.getElementById("stats").textContent =
    `${nodes.length} nodes · ${edges.length} edges`;

  const container = document.getElementById("network");

  const options = {
    physics: {
      enabled: true,
      barnesHut: {
        gravitationalConstant: -8000,
        centralGravity: 0.3,
        springLength: 160,
        springConstant: 0.04,
        damping: 0.09,
      },
      stabilization: { iterations: 150, fit: true },
    },
    interaction: {
      hover: true,
      tooltipDelay: 80,
      navigationButtons: true,
      keyboard: true,
    },
    layout: { improvedLayout: nodes.length < 300 },
  };

  if (network) {
    network.setData({ nodes, edges });
  } else {
    network = new vis.Network(container, { nodes, edges }, options);

    // Custom tooltip using absolute-positioned div
    const tooltip = document.getElementById("tooltip");
    network.on("hoverNode", params => {
      const node = network.body.nodes[params.node];
      if (!node) return;
      tooltip.innerHTML = node.options.title || "";
      tooltip.style.display = "block";
    });
    network.on("blurNode", () => { tooltip.style.display = "none"; });
    network.on("mousemove", params => {
      if (tooltip.style.display === "none") return;
      tooltip.style.left = (params.event.clientX + 14) + "px";
      tooltip.style.top  = (params.event.clientY - 10) + "px";
    });
  }
}

renderGraph();
</script>
</body>
</html>
"""


class HtmlVisualizer:
    """Standalone HTML + vis.js knowledge graph visualizer (no PyVis dependency)."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        filter_types: list[str] | None = None,
        min_weight: float = 0.0,
        max_nodes: int = 200,
    ):
        self.graph = graph
        self.filter_types = set(filter_types) if filter_types else None
        self.min_weight = min_weight
        self.max_nodes = max_nodes
        self._output_path: Path | None = None

    def generate_html(self) -> str:
        """Generate the standalone HTML string.

        Returns:
            Complete HTML document as a string
        """
        graph_data = self._build_graph_data()
        json_str = json.dumps(graph_data, ensure_ascii=False)
        return _HTML_TEMPLATE.replace("__GRAPH_DATA__", json_str)

    def save(self, path: str) -> None:
        """Save the visualization as a standalone HTML file.

        Args:
            path: Output file path (e.g. "graph.html")
        """
        self._output_path = Path(path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html()
        self._output_path.write_text(html, encoding="utf-8")
        print(f"Visualization saved to {self._output_path}")

    def show(self) -> None:
        """Open the visualization in the default web browser."""
        if not self._output_path:
            raise ValueError("Call save() before show()")
        webbrowser.open(f"file://{self._output_path.absolute()}")

    def _build_graph_data(self) -> dict:
        """Build serialisable graph data dict."""
        entities = {}
        for eid, entity in self.graph.entities.items():
            if self.filter_types and entity.entity_type not in self.filter_types:
                continue
            entities[eid] = {
                "entity_id": eid,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
            }

        entity_set = set(entities.keys())

        # Degree calculation for node-limit sorting (JS also does this, but
        # we include all data so JS can handle interactive filtering)
        degrees: dict[str, int] = defaultdict(int)
        relationships = []
        for rel in self.graph.relationships:
            if rel.weight < self.min_weight:
                continue
            if rel.source_entity_id not in entity_set or rel.target_entity_id not in entity_set:
                continue
            relationships.append({
                "relationship_id": rel.relationship_id,
                "source_entity_id": rel.source_entity_id,
                "target_entity_id": rel.target_entity_id,
                "relationship_type": rel.relationship_type,
                "description": rel.description,
                "weight": rel.weight,
            })
            degrees[rel.source_entity_id] += 1
            degrees[rel.target_entity_id] += 1

        # Trim to max_nodes by degree before embedding (reduces file size)
        if len(entities) > self.max_nodes:
            top_ids = sorted(entities, key=lambda eid: degrees[eid], reverse=True)[: self.max_nodes]
            entities = {eid: entities[eid] for eid in top_ids}
            trimmed_set = set(top_ids)
            relationships = [
                r for r in relationships
                if r["source_entity_id"] in trimmed_set and r["target_entity_id"] in trimmed_set
            ]

        return {"entities": entities, "relationships": relationships}
