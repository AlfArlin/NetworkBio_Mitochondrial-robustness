"""
panel_network_degree.py
Stand-alone figure: A = the intact network, B = the degree distribution.
Does not change any other script. Needs style.py and figures/network_layout.tsv
(made by network_layout.py) in the same folder.

Run the WHOLE file:   %run panel_network_degree.py
Saves figures/network_and_degree.pdf (vector, for Overleaf) + .png + .svg
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
importlib.reload(style)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EDGES_FILE = "mito_ppi_edges.tsv"
LAYOUT_FILE = "figures/network_layout.tsv"
OUT = "figures/network_and_degree"
SIDE_BY_SIDE = True            # False = A above B
WIDTHS = [2.2, 1]              # side by side: width of A relative to B
HEIGHT = 2.6                   # inches (side by side); stacked: height of B
NODE_SCALE = 1.0               # protein size in A (raise to 1.5-2 if too small)
DOT_SIZE = 22                  # dot area in B (pt^2)

style.apply()
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
edges = pd.read_csv(EDGES_FILE, sep="\t")[["source", "target"]].astype(str)
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
G.remove_edges_from(nx.selfloop_edges(G))
lay = pd.read_csv(LAYOUT_FILE, sep="\t")
lay["id"] = lay["id"].astype(str)
POS = {r.id: (r.x, -r.y) for r in lay.itertuples()}          # file uses Cytoscape's y-down
missing = [n for n in G if n not in POS]
if missing:
    raise SystemExit(f"{len(missing)} proteins are not in {LAYOUT_FILE}: run network_layout.py first.")


# ----------------------------------------------------------------------------
# PANELS
# ----------------------------------------------------------------------------
def draw_network(ax):
    xy = np.array([POS[n] for n in G])
    pad = 0.02 * np.ptp(xy, axis=0).max()
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.apply_aspect()
    pt_per_px = ax.get_window_extent().width * 72 / ax.figure.dpi / np.ptp(ax.get_xlim())
    d_pt = style.NETWORK_NODE_PX * pt_per_px * NODE_SCALE      # same proportions as in Cytoscape
    ax.add_collection(LineCollection([(POS[u], POS[v]) for u, v in G.edges()], colors=style.NETWORK_EDGE,
                                     alpha=min(1, style.NETWORK_EDGE_ALPHA + 0.2), linewidths=0.3, zorder=1))
    ax.scatter(xy[:, 0], xy[:, 1], s=d_pt ** 2, color=style.NETWORK_NODE, edgecolors="white",
               linewidths=0.08 * d_pt, zorder=2)


def draw_degree(ax):
    ks, counts = np.unique([d for _, d in G.degree()], return_counts=True)
    ax.scatter(ks, counts, s=DOT_SIZE, color=style.NETWORK_NODE, edgecolors="none", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.8, counts.max() * 1.5)
    ax.set_xlabel("Degree (k)", color=style.INK)
    ax.set_ylabel("Number of proteins", color=style.INK)
    ax.tick_params(colors=style.INK, labelcolor=style.INK, which="both")


if SIDE_BY_SIDE:
    fig = plt.figure(figsize=(style.FULL_WIDTH, HEIGHT))
    gs = fig.add_gridspec(1, 2, width_ratios=WIDTHS, wspace=0.08)
else:
    fig = plt.figure(figsize=(style.FULL_WIDTH, HEIGHT * 2.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.15)
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
draw_degree(ax_b)
draw_network(ax_a)
for ax, letter, x in [(ax_a, "A", 0.0), (ax_b, "B", -0.28)]:
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=style.PANEL_LABEL_SIZE,
            fontweight="bold", va="bottom", ha="left", color=style.INK)

style.save(fig, OUT)
print(f"Saved {OUT}.pdf / .png / .svg")
if "ipykernel" in sys.modules:
    plt.show()
