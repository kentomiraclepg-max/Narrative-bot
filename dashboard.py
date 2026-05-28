"""
Rich terminal dashboard for narrative screening results.
"""
from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.columns import Columns
from rich.text    import Text
from rich         import box

sys.path.insert(0, os.path.dirname(__file__))

console = Console()


def _sent_color(score: float) -> str:
    if score >  0.2: return "green"
    if score < -0.2: return "red"
    return "yellow"


def _sev_color(sev: str) -> str:
    return {"HIGH": "bold red", "MEDIUM": "bold yellow", "LOW": "dim"}.get(sev, "white")


def render(result: dict) -> None:
    from src.graph.queries  import top_entities_by_sentiment, graph_stats, most_connected
    from src.storage.db     import get_recent_articles, stats as db_stats

    console.clear()
    console.rule("[bold cyan]NARRATIVE SCREENING AGENT[/bold cyan]")

    # ── Stats bar ─────────────────────────────────────────────────────────────
    s = result.get("db_stats", {})
    console.print(
        f"  Articles 24h: [cyan]{s.get('articles_24h', 0)}[/]  "
        f"Total: [dim]{s.get('total_articles', 0)}[/]  "
        f"Narratives: [cyan]{s.get('total_narratives', 0)}[/]  "
        f"Alerts: [yellow]{s.get('total_alerts', 0)}[/]"
    )
    console.print()

    # ── Anomalies ─────────────────────────────────────────────────────────────
    anomalies = result.get("anomalies", [])
    if anomalies:
        t = Table(title="Detected Anomalies", box=box.SIMPLE_HEAD,
                  header_style="bold magenta")
        t.add_column("Type",     width=22)
        t.add_column("Topic",    width=20)
        t.add_column("Severity", width=8, justify="center")
        t.add_column("Detail",   width=40)

        for a in sorted(anomalies, key=lambda x: x.get("severity", ""), reverse=True):
            sev = a.get("severity", "")
            detail = ""
            if a["type"] == "volume_spike":
                detail = f"{a.get('article_count')} articles / {a.get('window_minutes')}min"
            elif a["type"] == "sentiment_spike":
                detail = f"Z={a.get('z_score',0):+.2f}  {a.get('direction','')}"
            elif a["type"] == "sentiment_divergence":
                detail = a.get("interpretation", "")[:40]

            t.add_row(
                a["type"],
                a.get("topic", "").replace("_", " ").upper(),
                f"[{_sev_color(sev)}]{sev}[/]",
                detail,
            )
        console.print(t)
    else:
        console.print("[dim]  No anomalies detected this scan.[/dim]")

    # ── Confirmed narratives ───────────────────────────────────────────────────
    confirmed = result.get("confirmed", [])
    if confirmed:
        console.print(Panel(
            "  ".join(f"[green]✓[/green] {c.replace('_',' ').upper()}" for c in confirmed),
            title="Cross-Verified Narratives"
        ))

    # ── Top sentiment entities ─────────────────────────────────────────────────
    try:
        top_ents = top_entities_by_sentiment(8)
        if top_ents:
            t2 = Table(title="Top Entities by Sentiment", box=box.SIMPLE_HEAD,
                       header_style="bold cyan")
            t2.add_column("Entity",   width=25)
            t2.add_column("Type",     width=12)
            t2.add_column("Mentions", width=8,  justify="right")
            t2.add_column("Avg Sent", width=10, justify="right")

            for e in top_ents:
                s_val = e["avg_sentiment"]
                t2.add_row(
                    e["label"],
                    e["type"],
                    str(e["mentions"]),
                    f"[{_sent_color(s_val)}]{s_val:+.3f}[/]",
                )
            console.print(t2)
    except Exception:
        pass

    # ── Recent articles ────────────────────────────────────────────────────────
    try:
        recent = get_recent_articles(hours=2, min_credibility=0.6)[:8]
        if recent:
            t3 = Table(title="Recent High-Credibility Articles (2h)",
                       box=box.SIMPLE_HEAD, header_style="bold white")
            t3.add_column("Source",    width=18)
            t3.add_column("Title",     width=50, overflow="fold")
            t3.add_column("Sentiment", width=10, justify="center")
            t3.add_column("Tone",      width=8,  justify="center")

            for a in recent:
                s_val = a.get("sentiment_score") or 0.0
                tone  = a.get("tone") or "—"
                tone_col = {"bullish": "green", "bearish": "red"}.get(tone, "dim")
                t3.add_row(
                    (a.get("source") or "")[:18],
                    (a.get("title") or "")[:50],
                    f"[{_sent_color(s_val)}]{s_val:+.3f}[/]",
                    f"[{tone_col}]{tone}[/]",
                )
            console.print(t3)
    except Exception:
        pass

    # ── Graph stats ────────────────────────────────────────────────────────────
    try:
        gs = graph_stats()
        console.print(
            f"\n  [dim]Knowledge Graph: {gs['nodes']} nodes · "
            f"{gs['edges']} edges · density={gs['density']}[/dim]"
        )
    except Exception:
        pass

    console.rule()


def show_narrative(topic: str) -> None:
    """Deep-dive display for a specific narrative topic."""
    from src.graph.queries import narrative_cluster, macro_impact_chain
    from src.storage.db    import get_sentiment_history

    console.print(Panel(f"[bold]{topic.upper().replace('_', ' ')}[/bold]",
                        title="Narrative Deep Dive"))

    cluster = narrative_cluster(topic)
    if cluster:
        t = Table(title="Associated Entities", box=box.SIMPLE_HEAD)
        t.add_column("Entity",   width=25)
        t.add_column("Type",     width=12)
        t.add_column("Weight",   width=8, justify="right")
        t.add_column("Mentions", width=8, justify="right")
        for e in cluster[:15]:
            t.add_row(e["entity"], e["type"], str(e["weight"]), str(e["mentions"]))
        console.print(t)

    chain = macro_impact_chain(topic)
    if chain:
        console.print(Panel(
            "\n".join(
                f"  Depth {k.split('_')[1]}: " + ", ".join(n["label"] for n in v[:5])
                for k, v in chain.items()
            ),
            title="Macro → Sector Impact Chain"
        ))


if __name__ == "__main__":
    # Quick test render with empty data
    render({
        "new_articles": 0, "processed": 0,
        "anomalies": [], "alerts": [],
        "confirmed": [], "verification": {},
        "db_stats": {"articles_24h": 0, "total_articles": 0,
                     "total_narratives": 0, "total_alerts": 0},
    })
