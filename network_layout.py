"""
network_layout.py
A clean layout of the complete (intact) mitochondrial PPI network, shared by
Cytoscape and every Python network figure.

Two ways to get the positions (LAYOUT in the CONFIG):

  "cytoscape"     keep the layout you made in Cytoscape (e.g. Prefuse Force Directed)
                  and only pull apart circles that sit on top of each other, so a
                  dense complex becomes a tight group of separate, countable circles
                  instead of one giant blob. Everything else stays where you put it.
  "kamada_kawai"  compute a new layout in Python (also "spring", "forceatlas2");
                  small components are then packed in tidy rows underneath.

Look: equal light-blue circles with a thin white rim, see-through light-grey
edges, no labels (size and colour stay free to encode something later).

Outputs (folder OUT_DIR):
  network_layout.tsv                 id, symbol, x, y, component  (Cytoscape coordinates, y down)
  cytoscape_positions_before.tsv     backup of the Cytoscape positions before they were changed
  network_plain.png / .pdf / .svg    the figure
Cytoscape (if open): the same positions and style applied to the network.

Run the WHOLE file:   %run network_layout.py   (notebook)   or   python network_layout.py
Run it BEFORE attack_views.py / adaptive_attack.py so their Cytoscape networks copy it.
"""
import os
import sys
import importlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import style
importlib.reload(style)               # always use the latest edits to style.py

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EDGES_FILE = "mito_ppi_edges.tsv"
GENES_FILE = "mito_genes.tsv"
OUT_DIR = "figures"
SEED = 42
LAYOUT = "cytoscape"           # "cytoscape" (keep your Cytoscape layout) | "kamada_kawai" | "spring" | "forceatlas2"
NODE_SPACING_PX = 30           # typical distance from a protein to its nearest neighbour (bigger = airier picture)
MIN_GAP_PX = 5                 # minimum white space between two circles (bigger = dense complexes spread more)
EDGE_PX = 40                   # Python layouts only: interaction length in the small components
COMPONENT_GAP_PX = 40          # Python layouts only: space between small components
ROW_GAP_PX = 70                # Python layouts only: space between the largest component and the rows below
USE_CYTOSCAPE = True
BASE_NETWORK_TITLE = "Yeast mitochondrial PPI"   # if not found, the network currently open in Cytoscape is used
STYLE_NAME = "Mito plain"

os.makedirs(OUT_DIR, exist_ok=True)
style.apply()
NODE_PX = style.NETWORK_NODE_PX
DMIN = NODE_PX + MIN_GAP_PX    # closest allowed distance between two circle centres

# ----------------------------------------------------------------------------
# 1. NETWORK
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
SYM = {}
if os.path.exists(GENES_FILE):
    _g = pd.read_csv(GENES_FILE, sep="\t").astype(str)
    SYM = dict(zip(_g["orf"], _g["symbol"]))
COMPS = sorted(nx.connected_components(G), key=len, reverse=True)
COMP_ID = {n: k for k, c in enumerate(COMPS, start=1) for n in c}      # 1 = largest

# ----------------------------------------------------------------------------
# 2. CYTOSCAPE CONNECTION
# ----------------------------------------------------------------------------
p4c, base = None, None
if USE_CYTOSCAPE or LAYOUT == "cytoscape":
    try:
        import py4cytoscape as p4c
        p4c.cytoscape_ping()
        if BASE_NETWORK_TITLE in p4c.get_network_list():
            base = p4c.get_network_suid(BASE_NETWORK_TITLE)
        elif LAYOUT == "cytoscape":
            base = p4c.get_network_suid()                     # the network open right now
            print(f"'{BASE_NETWORK_TITLE}' not found; using the open network '{p4c.get_network_name(base)}'")
    except Exception as e:
        p4c = None
        print("Cytoscape not reachable:", type(e).__name__, e)
if LAYOUT == "cytoscape" and base is None:
    raise SystemExit("LAYOUT = 'cytoscape' needs Cytoscape open with your network laid out.\n"
                     "Open it (or set LAYOUT = 'kamada_kawai') and run the file again.")


def cytoscape_names():
    """Name of every protein in Cytoscape (ORF, or gene symbol if the network was built with
    symbols), and the current Cytoscape positions."""
    cy = p4c.get_node_position(network=base)
    cy.index = cy.index.astype(str)
    names = {n: n for n in G if n in cy.index}
    if len(names) < 0.5 * len(G):
        by_symbol = {SYM.get(n, n): n for n in G}
        names = {by_symbol[c]: c for c in cy.index if c in by_symbol}
    return names, cy


# ----------------------------------------------------------------------------
# 3. LAYOUT
# ----------------------------------------------------------------------------
def remove_overlaps(xy, dmin, iterations=3000, seed=SEED):
    """Push apart any two circles closer than dmin. Each step moves a circle at most
    half a circle, so a collapsed complex unfolds into a tight disc instead of exploding."""
    rng = np.random.default_rng(seed)
    xy = xy + rng.normal(0, 0.01 * dmin, xy.shape)
    for _ in range(iterations):
        d = xy[:, None, :] - xy[None, :, :]
        dist = np.hypot(d[..., 0], d[..., 1])
        np.fill_diagonal(dist, np.inf)
        close = dist < dmin * 0.999
        if not close.any():
            break
        with np.errstate(invalid="ignore"):
            push = np.where(close, (dmin - dist) / np.maximum(dist, 1e-9) / 2, 0.0)
        step = (d * push[..., None]).sum(axis=1)
        size = np.hypot(step[:, 0], step[:, 1])[:, None]
        xy = xy + step * np.minimum(1.0, 0.5 * dmin / np.maximum(size, 1e-9))
    return xy


def nearest_neighbour_scale(xy, target):
    """Factor that makes the typical distance to the nearest protein equal to target."""
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    nn = nn[nn > 1e-6 * (np.ptp(xy) or 1)]          # ignore circles sitting exactly on top of each other
    return target / (np.median(nn) if len(nn) else 1.0)


def raw_layout(H):
    """Positions for one component (arbitrary units)."""
    n = H.number_of_nodes()
    if n == 1:
        return {next(iter(H)): np.zeros(2)}
    if n == 2:
        a, b = list(H)
        return {a: np.array([0.0, 0.0]), b: np.array([1.0, 0.0])}
    if n <= 30 or LAYOUT == "kamada_kawai":
        return {k: np.asarray(v) for k, v in nx.kamada_kawai_layout(H).items()}
    if LAYOUT == "forceatlas2" and hasattr(nx, "forceatlas2_layout"):
        return {k: np.asarray(v) for k, v in
                nx.forceatlas2_layout(H, max_iter=2000, seed=SEED, scaling_ratio=2.0, gravity=1.0).items()}
    return {k: np.asarray(v) for k, v in
            nx.spring_layout(H, seed=SEED, iterations=1000, k=1.6 / np.sqrt(n)).items()}


def python_layout():
    """New layout: largest component on top, small components packed in rows below."""
    pos = {}
    L = G.subgraph(COMPS[0])
    p = raw_layout(L)
    nodes = list(p)
    xy = np.array([p[n] for n in nodes], dtype=float)
    xy = remove_overlaps(xy * nearest_neighbour_scale(xy, NODE_SPACING_PX), DMIN)
    xy -= xy.min(axis=0)
    pos.update(zip(nodes, xy))
    width = max(xy[:, 0].max(), 600)
    y_next = xy[:, 1].max() + ROW_GAP_PX                    # Cytoscape: y grows downwards
    x, row_height = 0.0, 0.0
    for c in COMPS[1:]:
        H = G.subgraph(c)
        p = raw_layout(H)
        nodes = list(p)
        xy = np.array([p[n] for n in nodes], dtype=float)
        if H.number_of_edges():
            xy *= EDGE_PX / (np.median([np.hypot(*(p[u] - p[v])) for u, v in H.edges()]) or 1.0)
        xy = remove_overlaps(xy, DMIN)
        xy -= xy.min(axis=0)
        w, h = xy.max(axis=0)
        if x > 0 and x + w > width:                          # start a new row
            x, y_next, row_height = 0.0, y_next + row_height + COMPONENT_GAP_PX, 0.0
        pos.update((n, v + np.array([x, y_next])) for n, v in zip(nodes, xy))
        x += w + COMPONENT_GAP_PX
        row_height = max(row_height, h)
    return pos


def cytoscape_layout(names, cy):
    """Your Cytoscape layout, scaled to the circle size, with overlapping circles pulled apart."""
    missing = [n for n in G if n not in names]
    if missing:
        raise SystemExit(f"{len(missing)} proteins of {EDGES_FILE} are not in the Cytoscape network "
                         f"(e.g. {missing[:3]}). Check that it is the right network.")
    nodes = list(G)
    xy = cy.loc[[names[n] for n in nodes], ["x", "y"]].to_numpy(float)
    xy = xy * nearest_neighbour_scale(xy, NODE_SPACING_PX)  # same shape, only zoomed
    xy = remove_overlaps(xy, DMIN)
    xy -= xy.min(axis=0)
    return dict(zip(nodes, xy))


if base is not None:
    NAMES, CY_BEFORE = cytoscape_names()
    CY_BEFORE.to_csv(f"{OUT_DIR}/cytoscape_positions_before.tsv", sep="\t", index_label="name")
else:
    NAMES = {n: n for n in G}

pos = cytoscape_layout(NAMES, CY_BEFORE) if LAYOUT == "cytoscape" else python_layout()

table = pd.DataFrame({"id": list(G.nodes())})
table["symbol"] = table["id"].map(lambda n: SYM.get(n, n))
table["x"] = table["id"].map(lambda n: round(float(pos[n][0]), 2))
table["y"] = table["id"].map(lambda n: round(float(pos[n][1]), 2))
table["component"] = table["id"].map(COMP_ID)
table.to_csv(f"{OUT_DIR}/network_layout.tsv", sep="\t", index=False)
print(f"Layout ({LAYOUT}): {G.number_of_nodes()} proteins in {len(COMPS)} components "
      f"(largest: {len(COMPS[0])} proteins)")
print(f"Saved {OUT_DIR}/network_layout.tsv")

# ----------------------------------------------------------------------------
# 4. FIGURE (matches the Cytoscape view: same positions, sizes scaled alike)
# ----------------------------------------------------------------------------
P = {r.id: (r.x, -r.y) for r in table.itertuples()}         # y flipped for matplotlib
xs = np.array([p[0] for p in P.values()])
ys = np.array([p[1] for p in P.values()])
pad = NODE_PX
span_x, span_y = np.ptp(xs) + 2 * pad, np.ptp(ys) + 2 * pad
fig_w = style.FULL_WIDTH
fig, ax = plt.subplots(figsize=(fig_w, fig_w * span_y / span_x))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(xs.min() - pad, xs.max() + pad)
ax.set_ylim(ys.min() - pad, ys.max() + pad)
ax.set_aspect("equal")
ax.set_axis_off()
pt_per_px = fig_w * 72 / span_x                                # points per layout pixel
nx.draw_networkx_edges(G, P, ax=ax, edge_color=style.NETWORK_EDGE, alpha=style.NETWORK_EDGE_ALPHA,
                       width=max(0.25, 1.0 * pt_per_px))
nx.draw_networkx_nodes(G, P, ax=ax, node_color=style.NETWORK_NODE, node_size=(NODE_PX * pt_per_px) ** 2,
                       edgecolors=style.NETWORK_NODE_RIM, linewidths=style.NETWORK_NODE_RIM_PX * pt_per_px)
style.save(fig, f"{OUT_DIR}/network_plain")
print(f"Saved {OUT_DIR}/network_plain.png / .pdf / .svg")

# ----------------------------------------------------------------------------
# 5. CYTOSCAPE: same positions and look
# ----------------------------------------------------------------------------
if USE_CYTOSCAPE and p4c is not None:
    try:
        if base is None:                                     # no network yet: create it
            base = p4c.create_network_from_data_frames(
                nodes=table[["id", "symbol"]], edges=edges.assign(interaction="pp"),
                title=BASE_NETWORK_TITLE, collection="SGD GO:0005739")
        cy_table = table.assign(cy_name=table["id"].map(NAMES))
        p4c.load_table_data(cy_table[["cy_name", "symbol", "component"]], data_key_column="cy_name",
                            table_key_column="name", network=base)
        names = cy_table["cy_name"].tolist()
        # bypass=False: ordinary positions, so you can still drag nodes by hand
        p4c.set_node_property_bypass(names, table["x"].tolist(), "NODE_X_LOCATION", bypass=False, network=base)
        p4c.set_node_property_bypass(names, table["y"].tolist(), "NODE_Y_LOCATION", bypass=False, network=base)
        print("Cytoscape: positions applied")
    except Exception as e:
        print("Cytoscape: could not apply the positions:", type(e).__name__, e)
        base = None

    if base is not None:
        try:
            if STYLE_NAME in p4c.get_visual_style_names():
                p4c.delete_visual_style(STYLE_NAME)
            defaults = {"NODE_SHAPE": "ELLIPSE", "NODE_SIZE": NODE_PX, "NODE_FILL_COLOR": style.NETWORK_NODE,
                        "NODE_BORDER_WIDTH": style.NETWORK_NODE_RIM_PX, "NODE_BORDER_PAINT": style.NETWORK_NODE_RIM,
                        "NODE_TRANSPARENCY": 255, "NODE_LABEL_TRANSPARENCY": 0,
                        "EDGE_WIDTH": 1, "EDGE_STROKE_UNSELECTED_PAINT": style.NETWORK_EDGE,
                        "EDGE_TRANSPARENCY": int(round(255 * style.NETWORK_EDGE_ALPHA)),
                        "NETWORK_BACKGROUND_PAINT": "#FFFFFF"}
            p4c.create_visual_style(STYLE_NAME, defaults, [])          # no mappings: equal circles, no labels
            try:
                p4c.lock_node_dimensions(True, style_name=STYLE_NAME)
            except Exception:
                pass
            p4c.set_visual_style(STYLE_NAME, network=base)
            p4c.fit_content(network=base)
            p4c.export_image(os.path.abspath(f"{OUT_DIR}/cytoscape_network_plain.png"), type="PNG",
                             zoom=300, network=base, overwrite_file=True)
            print(f"Cytoscape: style '{STYLE_NAME}' applied; image saved in '{OUT_DIR}'")
        except Exception as e:
            print("Cytoscape: style step skipped:", type(e).__name__, e)

if "ipykernel" in sys.modules:
    plt.show()
