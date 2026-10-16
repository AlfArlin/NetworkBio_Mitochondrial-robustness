"""
panel_attacks.py
Stand-alone multi-panel figure of where the network breaks, for the report (vector PDF):

  A  Static betweenness     B  Static closeness     C  Static degree
  D  Adaptive betweenness   E  Adaptive closeness   F  Adaptive degree
  G  Random, highest damage H  Random, lowest damage    (9th cell: legend)
Each panel is labelled with its letter and the critical protein only.

Targeted panels: the same look as the split views (split_view.py), rebuilt from the saved
split tables. Random panels: the same colours, after removing RANDOM_FRACTION of the
proteins at random:
  light blue = still in the largest component, yellow = cut off from it,
  black = removed. Only proteins of the intact largest component are drawn,
  as in the targeted panels.

Does not change any other script. Needs, in the same folder: style.py, split_view.py,
mito_ppi_edges.tsv, attack_static/split_*.tsv, attack_adaptive/split_adaptive_*.tsv,
attack_random/random_runs.tsv  (made by the attack scripts and random_failure.py).

Run the WHOLE file:   %run panel_attacks.py
Saves figures/attack_panels.pdf (vector) + .png + .svg
"""
import os
import sys
import importlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import style
import split_view as sv
importlib.reload(style)
importlib.reload(sv)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EDGES_FILE = "mito_ppi_edges.tsv"
GENES_FILE = "mito_genes.tsv"
OUT = "figures/attack_panels"
PANELS = [("static", "betweenness"), ("static", "closeness"), ("static", "degree"),
          ("adaptive", "betweenness"), ("adaptive", "closeness"), ("adaptive", "degree"),
          ("random", "most"), ("random", "least")]       # the next cell holds the legend
COLUMNS = 3
FIG_WIDTH = style.FULL_WIDTH   # inches (6.3 = text width of an A4 page)
PANEL_HEIGHT = 2.2             # inches per row (panel + its letter)
SIZE = 1.8                     # protein / diamond size relative to the single split figures
TITLE_SIZE = 7                 # panel titles (pt)
TITLE_SIZE_LETTER = 9          # panel letters (pt)
LEGEND_SIZE = 7                # legend text (pt)
LEGEND = [                     # panel 9: (symbol, text); symbols: diamond, blue, yellow, black, dashed, yellow_line, grey_line
    ("diamond", "Critical protein"),
    ("blue", "Largest component"),
    ("yellow", "Split off / cut off"),
    ("black", "Removed"),
    ("dashed", "Interaction cut by the critical protein"),
    ("yellow_line", "Interaction in the split-off part"),
    ("grey_line", "Interaction in the largest component"),
]
RANDOM_FRACTION = 0.10         # random panels: fraction removed (same as random_failure.py's pictures)
RANDOM_SEED = 42               # must match SEED in random_failure.py
NAMES = {"betweenness": "betweenness", "closeness": "closeness", "degree": "degree"}
RANDOM_NAMES = {"most": "Random, highest damage", "least": "Random, lowest damage",
                "typical": "Random, typical run"}

style.apply()
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
SYM = {}
if os.path.exists(GENES_FILE):
    _g = pd.read_csv(GENES_FILE, sep="\t").astype(str)
    SYM = dict(zip(_g["orf"], _g["symbol"]))


def sym(n):
    return SYM.get(n, n)


POS = sv.make_layout(G)                     # same layout as the split views
LCC0 = max(nx.connected_components(G), key=len)


def targeted_view(mode, s):
    path = f"attack_static/split_{s}.tsv" if mode == "static" else f"attack_adaptive/split_adaptive_{s}.tsv"
    if not os.path.exists(path):
        print(f"  missing {path}: run the attack script first")
        return None, None
    t = pd.read_csv(path, sep="\t")
    t["id"] = t["id"].astype(str)
    shown = set(t["id"])
    removed = [n for n in G if n not in shown] + t.loc[t["fate"] == "previously removed", "id"].tolist()
    critical = t.loc[t["fate"] == "critical protein", "id"].iloc[0]
    return sv.build_view(G, removed, critical, sym), len(removed) + 1


def random_removed(which):
    """Proteins removed in the chosen random run, re-created exactly as random_failure.py does."""
    runs = pd.read_csv("attack_random/random_runs.tsv", sep="\t").sort_values("R_lcc").reset_index(drop=True)
    row = {"most": runs.iloc[0], "least": runs.iloc[-1], "typical": runs.iloc[len(runs) // 2]}[which]
    run_no = int(row["run"])
    nodes = list(G.nodes())
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(run_no):
        order = rng.permutation(nodes)
    k = max(1, round(RANDOM_FRACTION * len(nodes)))
    return list(order[:k]), run_no, float(row["R_lcc"])


def draw_random(ax, removed, scale):
    """Random-failure panel in the split-view colours."""
    gone = set(removed)
    H = G.subgraph(LCC0 - gone)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    largest = comps[0] if comps else set()
    cut = set(H) - largest
    shown = list(LCC0)
    P = {n: np.asarray(POS[n], float) for n in shown}
    xy = np.array(list(P.values()))
    pad = 0.04 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_anchor("N")
    for group, colour in [(largest, style.EDGE), (cut, style.SPLIT_EDGE)]:
        segs = [(P[u], P[v]) for u, v in H.subgraph(group).edges()]
        if segs:
            ax.add_collection(LineCollection(segs, colors=colour, linewidths=0.8 * scale, zorder=1))
    size = dict(zip(sv.FATES, sv.FATE_SIZES))
    for group, colour, s, border in [(largest, style.STAYS, size["stays in largest part"], False),
                                     (cut, style.SPLIT, size["splits off"], True),
                                     (gone & LCC0, style.REMOVED, size["previously removed"], False)]:
        nl = list(group)
        if nl:
            ax.scatter([P[n][0] for n in nl], [P[n][1] for n in nl], s=sv._mpl_size(s, scale), c=colour,
                       edgecolors="black" if border else colour, linewidths=0.5 * scale if border else 0, zorder=3)
    return len(LCC0), len(largest)


# ----------------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------------
def draw_legend(ax, scale):
    """Panel with every symbol used in the other panels."""
    from matplotlib.lines import Line2D
    size = dict(zip(sv.FATES, sv.FATE_SIZES))
    ms = lambda f: np.sqrt(sv._mpl_size(size[f], scale))
    look = {
        "diamond": Line2D([], [], ls="", marker="D", ms=ms("critical protein") * 0.75, mfc=style.CRITICAL,
                          mec="black", mew=1.0),
        "blue": Line2D([], [], ls="", marker="o", ms=ms("stays in largest part"), mfc=style.STAYS, mec=style.STAYS),
        "yellow": Line2D([], [], ls="", marker="o", ms=ms("splits off"), mfc=style.SPLIT, mec="black", mew=0.5),
        "black": Line2D([], [], ls="", marker="o", ms=ms("previously removed"), mfc=style.REMOVED, mec=style.REMOVED),
        "dashed": Line2D([], [], color=style.INK, lw=0.9, ls=(0, (4, 2.5))),
        "yellow_line": Line2D([], [], color=style.SPLIT_EDGE, lw=1.0),
        "grey_line": Line2D([], [], color=style.EDGE, lw=1.0),
    }
    ax.set_axis_off()
    ax.legend([look[k] for k, _ in LEGEND], [t for _, t in LEGEND], loc="center left", frameon=False,
              fontsize=LEGEND_SIZE, handlelength=2.2, handletextpad=0.8, labelspacing=1.1, borderaxespad=0)


nrow = int(np.ceil((len(PANELS) + 1) / COLUMNS))
fig, axes = plt.subplots(nrow, COLUMNS, figsize=(FIG_WIDTH, nrow * PANEL_HEIGHT), squeeze=False)
scale = FIG_WIDTH / COLUMNS / 10 * SIZE
cells = list(axes.flat)
for ax, (mode, s), letter in zip(cells, PANELS, "ABCDEFGHIJKL"):
    if mode == "random":
        removed, run_no, R = random_removed(s)
        before, after = draw_random(ax, removed, scale)
        name = ""
        print(f"  {letter}: random ({s}) run {run_no}, {len(removed)} removed: {before} -> {after}")
    else:
        v, step = targeted_view(mode, s)
        if v is None:
            ax.set_axis_off()
            continue
        sv.draw_panel(ax, v, POS, scale=scale)
        name = v["label"][v["critical"]]
        print(f"  {letter}: {mode} {s}: {name}, step {step}: {v['lcc_before']} -> {v['lcc_after']}")
    tag = ax.text(0, 1.0, letter, transform=ax.transAxes, fontsize=TITLE_SIZE_LETTER, fontweight="bold",
                  va="baseline", ha="left", color=style.INK)
    if name:                                   # right after the letter, never overlapping it
        ax.annotate(name, xy=(1, 0), xycoords=tag, xytext=(4, 0), textcoords="offset points",
                    fontsize=TITLE_SIZE, va="bottom", ha="left", color=style.INK)
draw_legend(cells[len(PANELS)], scale)
for ax in cells[len(PANELS) + 1:]:
    ax.set_axis_off()
fig.tight_layout(h_pad=0.8, w_pad=0.6)
style.save(fig, OUT)
print(f"Saved {OUT}.pdf / .png / .svg")
if "ipykernel" in sys.modules:
    plt.show()
