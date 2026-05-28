"""
Knowledge graph queries: entity lookup, macro impact chains, top connected nodes.
"""
from __future__ import annotations

import networkx as nx

from src.graph.builder import _get_graph


def get_entity(name: str) -> dict | None:
    g     = _get_graph()
    node  = f"COMPANY::{name.lower()}"
    if not g.has_node(node):
        # Try partial match
        for n in g.nodes:
            if name.lower() in n.lower():
                node = n
                break
        else:
            return None

    data = dict(g.nodes[node])
    neighbors = list(g.neighbors(node))
    data["connections"] = len(neighbors)
    avg_sent = (data.get("sentiment_sum", 0.0) / data.get("mention_count", 1)
                if data.get("mention_count") else 0.0)
    data["avg_sentiment"] = round(avg_sent, 4)
    return data


def macro_impact_chain(theme: str, depth: int = 3) -> dict:
    """
    Traverse: MACRO_THEME → SECTOR → COMPANY
    Returns dict of impacted nodes at each depth.
    """
    g = _get_graph()
    if not g.has_node(theme):
        return {}

    result = {}
    frontier = {theme}
    for level in range(1, depth + 1):
        next_frontier = set()
        for node in frontier:
            for neighbor in g.neighbors(node):
                if neighbor not in result and neighbor != theme:
                    next_frontier.add(neighbor)

        if not next_frontier:
            break

        result[f"depth_{level}"] = [
            {
                "node":      n,
                "type":      g.nodes[n].get("type", ""),
                "label":     g.nodes[n].get("label", n),
                "edge_weight": g[f]["n"]["weight"] if g.has_edge(f, n) else 0
                               if False else sum(
                    g[p][n].get("weight", 0) for p in g.predecessors(n) if p in frontier
                ),
            }
            for n in next_frontier
        ]
        frontier = next_frontier

    return result


def top_entities_by_sentiment(n: int = 10, entity_type: str | None = None) -> list[dict]:
    """Return top N entities by mention-weighted sentiment."""
    g = _get_graph()
    rows = []
    for node, data in g.nodes(data=True):
        if entity_type and data.get("type") != entity_type:
            continue
        mc = data.get("mention_count", 0)
        if mc < 2:
            continue
        avg_sent = data.get("sentiment_sum", 0.0) / mc
        rows.append({
            "node":         node,
            "label":        data.get("label", node),
            "type":         data.get("type", ""),
            "mentions":     mc,
            "avg_sentiment": round(avg_sent, 4),
        })

    rows.sort(key=lambda x: abs(x["avg_sentiment"]) * x["mentions"], reverse=True)
    return rows[:n]


def most_connected(n: int = 10) -> list[dict]:
    """Return N most connected nodes (by degree)."""
    g = _get_graph()
    degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:n]
    return [
        {
            "node":   node,
            "label":  g.nodes[node].get("label", node),
            "type":   g.nodes[node].get("type", ""),
            "degree": deg,
        }
        for node, deg in degrees
    ]


def narrative_cluster(topic: str) -> list[dict]:
    """Find all entities strongly associated with a narrative topic."""
    g = _get_graph()
    if not g.has_node(topic):
        return []

    # Get predecessors (entities pointing TO this topic)
    preds = list(g.predecessors(topic))
    result = []
    for p in preds:
        data = g.nodes[p]
        edge = g[p][topic]
        result.append({
            "entity":  data.get("label", p),
            "type":    data.get("type", ""),
            "weight":  edge.get("weight", 1),
            "mentions": data.get("mention_count", 0),
        })

    result.sort(key=lambda x: x["weight"], reverse=True)
    return result


def graph_stats() -> dict:
    g = _get_graph()
    type_counts: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        t = data.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "nodes":       g.number_of_nodes(),
        "edges":       g.number_of_edges(),
        "node_types":  type_counts,
        "density":     round(nx.density(g), 6),
    }
