"""Generate the link-jumping mechanism figure for the final report.

Output: docs/figures/fig_link_jumping.png

Layered diagram:
  layer 0 (left)   : the query (single node)
  layer 1          : 3 cosine seed chunks (top-K=3)
  layer 2 (middle) : entities collected from seed-chunk triples
  layer 3          : expanded chunks reachable through any entity
  layer 4 (right)  : final top-5 (subset of layer-1 ∪ layer-3)

Edges between layer 2 and layer 3 are dotted if a triple's temporal
validity is disjoint from the query year-set (so L3 cannot traverse
them).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "docs" / "figures" / "fig_link_jumping.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, text, color="#dae8fc", w=1.6, h=0.55, bold=False, edge="#6c8ebf"):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor=edge, linewidth=1.0,
        ))
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8.5, fontweight=weight, family="sans-serif")

    def arrow(x1, y1, x2, y2, dotted=False, color="#444"):
        ls = (0, (3, 3)) if dotted else "-"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=0.9,
                            linestyle=ls, alpha=0.85),
        )

    # Layer headers
    headers = [(1.0, 5.2, "(a) Query"),
               (3.0, 5.2, "(b) Cosine seeds (k=3)"),
               (5.5, 5.2, "(c) Entities from seed triples"),
               (8.0, 5.2, "(d) Expanded chunk pool"),
               (10.3, 5.2, "(e) Final top-5")]
    for x, y, t in headers:
        ax.text(x, y, t, ha="center", va="center", fontsize=9,
                fontweight="bold", color="#222")

    # (a) Query node
    box(1.0, 3.0, "Compare\nAAPL Services\nvs\nMSFT cloud\n(FY2022)",
        color="#fff2cc", edge="#d6b656", w=1.7, h=1.7)

    # (b) Seeds
    seeds = [
        (3.0, 4.2, "AAPL_FY22\nitem 7",  "#dae8fc"),
        (3.0, 3.0, "AAPL_FY22\nitem 8",  "#dae8fc"),
        (3.0, 1.8, "MSFT_FY22\nitem 7",  "#dae8fc"),
    ]
    for x, y, t, c in seeds:
        box(x, y, t, color=c, w=1.5, h=0.85)
        arrow(1.0 + 0.85, 3.0, x - 0.75, y, color="#1f77b4")

    # (c) Entities
    entities = [
        (5.5, 4.5, "Services revenue",       "#e1d5e7"),
        (5.5, 3.7, "Productivity & BP",      "#e1d5e7"),
        (5.5, 2.9, "Apple Inc.",             "#e1d5e7"),
        (5.5, 2.1, "Microsoft Corp.",        "#e1d5e7"),
        (5.5, 1.3, "fiscal 2022",            "#e1d5e7"),
    ]
    for x, y, t, c in entities:
        box(x, y, t, color=c, edge="#9673a6", w=1.6, h=0.5)
    # seed -> entity edges (each seed contributes a few)
    seed_entity_edges = [
        (0, 0), (0, 2), (0, 4),  # AAPL_item7 → Services, Apple, fiscal2022
        (1, 0), (1, 2),          # AAPL_item8 → Services, Apple
        (2, 1), (2, 3), (2, 4),  # MSFT_item7 → Productivity, MSFT, fiscal2022
    ]
    for s, e in seed_entity_edges:
        sx, sy, _, _ = seeds[s]
        ex, ey, _, _ = entities[e]
        arrow(sx + 0.75, sy, ex - 0.8, ey, color="#888")

    # (d) Expanded chunks
    expanded = [
        (8.0, 4.6, "AAPL_FY24\nitem 8",   "#d5e8d4", False),
        (8.0, 3.7, "AAPL_FY22\nitem 1",   "#d5e8d4", False),
        (8.0, 2.8, "MSFT_FY22\nitem 8",   "#d5e8d4", False),
        (8.0, 1.9, "AAPL_FY18\nitem 7",   "#fff4e5", True),   # disjoint validity
        (8.0, 1.0, "MSFT_FY24\nitem 7",   "#fff4e5", True),   # disjoint validity
    ]
    for x, y, t, c, _ in expanded:
        edge = "#82b366" if c == "#d5e8d4" else "#d79b00"
        box(x, y, t, color=c, edge=edge, w=1.5, h=0.85)
    # entity -> expanded edges (some dotted = L3-blocked)
    ent_exp_edges = [
        (0, 0, False), (0, 2, False),
        (1, 2, False),
        (2, 1, False), (2, 3, True),
        (3, 2, False), (3, 4, True),
        (4, 0, False), (4, 1, False),
    ]
    for e, x, dotted in ent_exp_edges:
        ex, ey, _, _ = entities[e]
        xx, xy_, _, _, _ = expanded[x]
        arrow(ex + 0.8, ey, xx - 0.75, xy_, dotted=dotted,
              color="#d79b00" if dotted else "#888")

    # (e) Final top-5
    final = [
        (10.3, 4.4, "AAPL_FY22\nitem 7",  "#1f77b4", "rank 1"),
        (10.3, 3.5, "AAPL_FY22\nitem 8",  "#1f77b4", "rank 2"),
        (10.3, 2.6, "MSFT_FY22\nitem 7",  "#1f77b4", "rank 3"),
        (10.3, 1.7, "MSFT_FY22\nitem 8",  "#82b366", "rank 4 (KG)"),
        (10.3, 0.8, "AAPL_FY24\nitem 8",  "#82b366", "rank 5 (KG)"),
    ]
    for x, y, t, edge, _ in final:
        ax.add_patch(FancyBboxPatch(
            (x - 0.85, y - 0.45), 1.7, 0.9,
            boxstyle="round,pad=0.04",
            facecolor="#ffffff", edgecolor=edge, linewidth=1.4,
        ))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5)

    # Pre-final → final arrow band
    for src_y in [4.6, 3.7, 2.8, 4.2, 3.0, 1.8]:
        # not perfect, just a hint
        pass

    # Legend
    leg_y = 0.25
    ax.text(0.7, leg_y, "─── traversable", color="#444", fontsize=8.5, va="center")
    ax.text(3.4, leg_y, "··· blocked under L3 (disjoint temporal validity)",
            color="#d79b00", fontsize=8.5, va="center")
    ax.text(7.6, leg_y, "Blue: cosine seed   Green: KG-expanded",
            fontsize=8.5, va="center")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
