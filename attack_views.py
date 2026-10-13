"""
attack_views.py
Shows WHERE the network breaks in each static attack (degree, betweenness, closeness).

For each strategy it finds the single most damaging removal (the "critical protein"),
then shows the network just before that removal, coloured by what happens next:
    critical protein       -> the one being removed
    stays in largest part  -> remains in the largest connected component
    splits off             -> gets cut away from the largest component
    already disconnected   -> was already separated by earlier removals
Proteins removed earlier in the attack are not shown in that view.

Outputs (folder OUT_DIR): split_<strategy>.png, split_<strategy>.tsv
Cytoscape (if open)     : one network per strategy in the collection "Static attack views"

Run the WHOLE file:   %run attack_views.py    (notebook)   or   python attack_views.py
"""
import os
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import importlib
import split_view as sv            # shared look + drawing (keep split_view.py in this folder)
importlib.reload(sv)               # always use the latest edits to split_view.py

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EDGES_FILE = "mito_ppi_edges.tsv"
GENES_FILE = "mito_genes.tsv"
OUT_DIR = "attack_static"
STRATEGIES = ["degree", "betweenness", "closeness"]
STEP_OVERRIDE = {}            # e.g. {"degree": 90} to look at a step of your choice
USE_CYTOSCAPE = True
BASE_NETWORK_TITLE = "Yeast mitochondrial PPI"     # used only to copy its layout
COLLECTION = "Static attack views"
# Colours, sizes, shapes, labels and edge styles live in split_view.py (LOOK block).

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. NETWORK AND STATIC RANKINGS (same rules as static_attack.py)
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
N = G.number_of_nodes()

SYM = {}
if os.path.exists(GENES_FILE):
    _g = pd.read_csv(GENES_FILE, sep="\t").astype(str)
    SYM = dict(zip(_g["orf"], _g["symbol"]))


def sym(n):
    return SYM.get(n, n)


scores = {"degree": dict(G.degree()),
          "betweenness": nx.betweenness_centrality(G),
          "closeness": nx.closeness_centrality(G)}
orders = {s: sorted(G.nodes(), key=lambda n: (-scores[s][n], n)) for s in STRATEGIES}


def largest_component(H):
    return max(nx.connected_components(H), key=len, default=set())


def lcc_sizes(order):
    """Size of the largest component after 0, 1, 2, ... removals."""
    H = G.copy()
    sizes = [len(largest_component(H))]
    for n in order:
        H.remove_node(n)
        sizes.append(len(largest_component(H)))
    return sizes


# ----------------------------------------------------------------------------
# 2. FIND THE SPLIT FOR EACH STRATEGY
# ----------------------------------------------------------------------------
views = {}
for s in STRATEGIES:
    sizes = lcc_sizes(orders[s])
    drops = [sizes[i - 1] - sizes[i] for i in range(1, len(sizes))]
    step = STEP_OVERRIDE.get(s, drops.index(max(drops)) + 1)     # 1-based step number
    v = sv.build_view(G, orders[s][:step - 1], orders[s][step - 1], sym)
    v["step"] = step
    views[s] = v
    v["nodes"].to_csv(f"{OUT_DIR}/split_{s}.tsv", sep="\t", index=False)

    print(f"\n=== {s.upper()} attack: most damaging removal is step {step}, {sym(v['critical'])} ===")
    print(f"Largest component: {v['lcc_before']} -> {v['lcc_after']} proteins")
    print(f"Pieces that split off: {[len(p) for p in v['pieces'][1:]] or 'none'}")
    print(f"Removed before it ({len(v['removed_before'])}): {', '.join(sym(n) for n in v['removed_before'])}")
    for i, p in enumerate(v["pieces"][1:4], start=1):
        print(f"Piece {i} ({len(p)}): {', '.join(sorted(sym(n) for n in p))}")

# ----------------------------------------------------------------------------
# 3. FIGURES (no Cytoscape needed)
# ----------------------------------------------------------------------------
pos = sv.make_layout(G)
for s, v in views.items():
    sv.draw_view(v, pos, v["step"], f"static {s}", f"{OUT_DIR}/split_{s}.png")
print(f"\nFigures and tables saved in '{OUT_DIR}': split_<strategy>.png / .tsv")

# ----------------------------------------------------------------------------
# 4. CYTOSCAPE
# ----------------------------------------------------------------------------
if USE_CYTOSCAPE:
    sv.send_to_cytoscape(
        [(f"Split by {s} attack (step {v['step']}, {sym(v['critical'])})", v, f"cytoscape_split_{s}.png")
         for s, v in views.items()],
        COLLECTION, BASE_NETWORK_TITLE, OUT_DIR)

if "ipykernel" in sys.modules:
    plt.show()
