"""
Knowledge graph builder using NetworkX.
Nodes: entities (company, person, location, theme)
Edges: co-occurrence in articles, supply chain links, macro→sector→stock
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any

import networkx as nx

import config


_graph: nx.DiGraph | None = None


def _get_graph() -> nx.DiGraph:
    global _graph
    if _graph is None:
        _graph = _load_or_create()
    return _graph


def _load_or_create() -> nx.DiGraph:
    os.makedirs(os.path.dirname(config.GRAPH_PATH), exist_ok=True)
    if os.path.exists(config.GRAPH_PATH):
        try:
            with open(config.GRAPH_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    g = nx.DiGraph()
    _seed_macro_structure(g)
    return g


def _seed_macro_structure(g: nx.DiGraph) -> None:
    """Pre-wire known macro→sector→stock relationships."""
    macro_to_sector = {
        "rate_hike":    ["Banking", "Insurance", "Financial Services"],
        "rate_cut":     ["Real Estate", "Utilities", "Consumer Discretionary"],
        "inflation":    ["Energy", "Materials", "Consumer Staples"],
        "recession":    ["Consumer Discretionary", "Industrials"],
        "supply_chain": ["Technology", "Automotive", "Semiconductor"],
        "bi_rate":      ["Perbankan IDX", "Properti IDX"],
        "rupiah":       ["Eksportir IDX", "Importir IDX"],
        "commodity":    ["Energi IDX", "Mining IDX"],
    }
    for macro, sectors in macro_to_sector.items():
        g.add_node(macro, type="MACRO_THEME", label=macro)
        for sector in sectors:
            g.add_node(sector, type="SECTOR", label=sector)
            g.add_edge(macro, sector, relation="IMPACTS", weight=1.0)


def save_graph() -> None:
    g = _get_graph()
    with open(config.GRAPH_PATH, "wb") as f:
        pickle.dump(g, f)


def add_article_entities(article_id: str, entities: list[dict],
                          themes: list[str], sentiment: float) -> None:
    """
    Add entity nodes and edges from a processed article.
    Edges represent co-occurrence + sentiment-weighted relationship.
    """
    g = _get_graph()
    ts = datetime.utcnow().isoformat()

    entity_nodes = []
    for ent in entities:
        node_id = f"{ent['category']}::{ent['text'].lower()}"
        if not g.has_node(node_id):
            g.add_node(node_id, type=ent["category"], label=ent["text"],
                       first_seen=ts, mention_count=0, sentiment_sum=0.0)

        data = g.nodes[node_id]
        data["mention_count"] = data.get("mention_count", 0) + 1
        data["sentiment_sum"] = data.get("sentiment_sum", 0.0) + sentiment
        data["last_seen"]     = ts
        entity_nodes.append(node_id)

    # Co-occurrence edges between entities in same article
    for i in range(len(entity_nodes)):
        for j in range(i + 1, len(entity_nodes)):
            src, dst = entity_nodes[i], entity_nodes[j]
            if g.has_edge(src, dst):
                g[src][dst]["weight"]  = g[src][dst].get("weight", 0) + 1
                g[src][dst]["last_seen"] = ts
            else:
                g.add_edge(src, dst, relation="CO_OCCURS",
                           weight=1, sentiment=sentiment, last_seen=ts)

    # Link entities to macro themes
    for theme in themes:
        theme_node = theme
        if not g.has_node(theme_node):
            g.add_node(theme_node, type="MACRO_THEME", label=theme)
        for ent_node in entity_nodes:
            if not g.has_edge(ent_node, theme_node):
                g.add_edge(ent_node, theme_node, relation="RELATED_TO",
                           weight=1, last_seen=ts)
            else:
                g[ent_node][theme_node]["weight"] = \
                    g[ent_node][theme_node].get("weight", 0) + 1


def update_from_articles(articles: list[dict]) -> None:
    for article in articles:
        entities = article.get("entities", [])
        themes   = article.get("macro_themes", [])
        score    = article.get("sentiment_score", 0.0)
        if isinstance(entities, str):
            import json
            try:
                entities = json.loads(entities)
            except Exception:
                entities = []
        add_article_entities(article["id"], entities, themes, score)

    save_graph()
    g = _get_graph()
    print(f"[Graph] Updated: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
