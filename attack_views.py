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
STYLE_NAME = "Attack split"

FATES = ["critical protein", "stays in largest part", "splits off", "already disconnected"]
FATE_COLORS = ["#D55E00", "#0072B2", "#E69F00", "#BBBBBB"]
FATE_SIZES = [70, 35, 35, 25]

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
    removed_before = orders[s][:step - 1]
    critical = orders[s][step - 1]

    before = G.copy()
    before.remove_nodes_from(removed_before)        # network just before the critical removal
    main = largest_component(before)
    after_main = before.subgraph(main - {critical})
    pieces = sorted(nx.connected_components(after_main), key=len, reverse=True)

    fate, piece = {}, {}
    for n in before:
        fate[n], piece[n] = "already disconnected", ""
    for i, p in enumerate(pieces):
        for n in p:
            fate[n] = "stays in largest part" if i == 0 else "splits off"
            piece[n] = "" if i == 0 else f"piece {i} ({len(p)} proteins)"
    fate[critical] = "critical protein"

    table = pd.DataFrame({"id": list(before.nodes())})
    table["symbol"] = table["id"].map(sym)
    table["fate"] = table["id"].map(fate)
    table["piece"] = table["id"].map(piece)
    table["degree_intact"] = table["id"].map(scores["degree"])
    table.to_csv(f"{OUT_DIR}/split_{s}.tsv", sep="\t", index=False)

    views[s] = dict(step=step, critical=critical, removed_before=removed_before,
                    before=before, table=table, pieces=pieces, fate=fate,
                    lcc_before=len(main), lcc_after=sizes[step])

    print(f"\n=== {s.upper()} attack: most damaging removal is step {step}, {sym(critical)} ===")
    print(f"Largest component: {len(main)} -> {sizes[step]} proteins")
    print(f"Pieces that split off: {[len(p) for p in pieces[1:]] or 'none'}")
    print(f"Removed before it ({len(removed_before)}): {', '.join(sym(n) for n in removed_before)}")
    for i, p in enumerate(pieces[1:4], start=1):
        print(f"Piece {i} ({len(p)}): {', '.join(sorted(sym(n) for n in p))}")

# ----------------------------------------------------------------------------
# 3. FIGURES (no Cytoscape needed)
# ----------------------------------------------------------------------------
def make_layout(G):
    """Largest component drawn by path distance; small components in rows underneath."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = nx.kamada_kawai_layout(G.subgraph(comps[0]))                 # coordinates roughly in [-1, 1]
    per_row, dx, dy = 12, 2.0 / 11, 0.16
    for i, c in enumerate(comps[1:]):
        cx, cy = -1 + (i % per_row) * dx, -1.25 - (i // per_row) * dy
        sub = nx.circular_layout(G.subgraph(c), scale=0.035 + 0.008 * len(c))
        for n, (x, y) in sub.items():
            pos[n] = (cx + x, cy + y)
    return pos


pos = make_layout(G)
color_of = dict(zip(FATES, FATE_COLORS))

for s, v in views.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # left: intact network, attack targets marked
    gone = set(v["removed_before"])
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="0.75", width=0.4, alpha=0.5)
    rest = [n for n in G if n not in gone and n != v["critical"]]
    nx.draw_networkx_nodes(G, pos, nodelist=rest, node_color="#9ecae1", node_size=22, ax=ax1, linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=list(gone), node_color="black", node_size=30, ax=ax1,
                           label=f"removed earlier ({len(gone)})")
    nx.draw_networkx_nodes(G, pos, nodelist=[v["critical"]], node_color=color_of["critical protein"],
                           node_size=140, ax=ax1, label=f"critical protein: {sym(v['critical'])}")
    ax1.set_title(f"Intact network ({N} proteins) with the first {v['step']} {s} targets marked")
    ax1.legend(loc="upper left", fontsize=9)

    # right: just before the critical removal, coloured by fate
    B = v["before"]
    crit_edges = list(B.edges(v["critical"]))
    other_edges = [e for e in B.edges() if v["critical"] not in e]
    nx.draw_networkx_edges(B, pos, edgelist=other_edges, ax=ax2, edge_color="0.7", width=0.4, alpha=0.6)
    nx.draw_networkx_edges(B, pos, edgelist=crit_edges, ax=ax2, edge_color=color_of["critical protein"],
                           width=1.6, style="dashed")
    for f in FATES[::-1]:
        nl = [n for n in B if v["fate"][n] == f]
        nx.draw_networkx_nodes(B, pos, nodelist=nl, node_color=color_of[f], ax=ax2, linewidths=0,
                               node_size=140 if f == "critical protein" else 26, label=f"{f} ({len(nl)})")
    # label the critical protein (and its direct partners, if few)
    lab = {v["critical"]: sym(v["critical"])}
    if B.degree(v["critical"]) <= 15:                 # partner names only when they fit
        lab.update({n: sym(n) for n in B.neighbors(v["critical"])})
    nx.draw_networkx_labels(B, pos, labels=lab, font_size=8, font_weight="bold", ax=ax2)
    ax2.set_title(f"Removing {sym(v['critical'])} (step {v['step']}): largest component "
                  f"{v['lcc_before']} -> {v['lcc_after']} proteins")
    ax2.legend(loc="upper left", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_axis_off()
    fig.suptitle(f"Static {s} attack: where the network breaks", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/split_{s}.png", dpi=200)

print(f"\nFigures and tables saved in '{OUT_DIR}': split_<strategy>.png / .tsv")

# ----------------------------------------------------------------------------
# 4. CYTOSCAPE
# ----------------------------------------------------------------------------
if USE_CYTOSCAPE:
    try:
        import py4cytoscape as p4c
        p4c.cytoscape_ping()
        connected = True
    except Exception as e:
        connected = False
        print("\nCytoscape not reachable, skipping it:", type(e).__name__, e)

    if connected:
        existing = p4c.get_network_list()
        base = p4c.get_network_suid(BASE_NETWORK_TITLE) if BASE_NETWORK_TITLE in existing else None
        style_ready = False
        made = []

        for s, v in views.items():
            title = f"Split by {s} attack (step {v['step']}, {sym(v['critical'])})"
            try:
                if title in p4c.get_network_list():
                    p4c.delete_network(title)
                B = v["before"]
                e_df = pd.DataFrame([(str(a), str(b)) for a, b in B.edges()], columns=["source", "target"])
                e_df["interaction"] = "pp"
                suid = p4c.create_network_from_data_frames(
                    nodes=v["table"].copy(), edges=e_df, title=title, collection=COLLECTION)
                made.append(title)
                print(f"Cytoscape: created '{title}'")
            except Exception as e:
                print(f"Cytoscape: could not create the {s} network:", type(e).__name__, e)
                continue

            try:        # same node positions as your original network, so views are comparable
                if base is not None:
                    p4c.layout_copycat(base, suid)
                else:
                    p4c.layout_network("force-directed", network=suid)
            except Exception as e:
                print("  layout step skipped:", type(e).__name__, e)

            try:        # one shared style: colour and size by fate, label by gene symbol
                if not style_ready:
                    if STYLE_NAME in p4c.get_visual_style_names():
                        p4c.delete_visual_style(STYLE_NAME)
                    defaults = {"NODE_SHAPE": "ELLIPSE", "NODE_SIZE": 35, "NODE_LABEL_FONT_SIZE": 9,
                                "NODE_BORDER_WIDTH": 0, "EDGE_TRANSPARENCY": 110, "EDGE_WIDTH": 1}
                    mappings = [
                        p4c.map_visual_property("node label", "symbol", "p", network=suid),
                        p4c.map_visual_property("node fill color", "fate", "d", FATES, FATE_COLORS, network=suid),
                        p4c.map_visual_property("node size", "fate", "d", FATES, FATE_SIZES, network=suid),
                    ]
                    p4c.create_visual_style(STYLE_NAME, defaults, mappings)
                    style_ready = True
                p4c.set_visual_style(STYLE_NAME, network=suid)
                p4c.fit_content(network=suid)
            except Exception as e:
                print("  style step skipped:", type(e).__name__, e)

            try:
                png = os.path.abspath(os.path.join(OUT_DIR, f"cytoscape_split_{s}.png"))
                p4c.export_image(png, type="PNG", network=suid, overwrite_file=True)
            except Exception as e:
                print("  image export skipped:", type(e).__name__, e)

        if made:
            print(f"\nCytoscape: look in the Network panel (left side) under '{COLLECTION}'.")
            print("Click a network name there to open its view. Your original network is unchanged.")

if "ipykernel" in sys.modules:
    plt.show()
